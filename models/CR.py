#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import tqdm
from torch.nn import Dropout

def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs))  # [bs]
    loss = torch.mean(loss)

    return loss

def l2_regularizer(weight, lambda_l2):
    return lambda_l2 * torch.norm(weight, 2)

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio])
    values = mask * values
    return values


def time_embedding(sequence, feature):
    mask = (sequence != 0).float()
    sequence_embedding = F.embedding(sequence.cuda(), feature.cuda(), padding_idx=0)
    return mask, sequence_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class diy_attention(nn.Module):
    def __init__(self, d_model, dropout=0.3) -> None:
        super(diy_attention, self).__init__()

        self.d_model = d_model
        self.dropout = Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.PositionalEncoding = PositionalEncoding(d_model, 50)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):

        x = self.PositionalEncoding(x)
        x = x.float()

        q, k, v, mask = x, x, x, mask

        # k = k + positional_encoding(k, self.d_model)  # it seems that result of removing positional_encoding is better?
        # q = q + positional_encoding(q, self.d_model)

        batch, time, dimension = q.shape

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        expanded_mask = mask
        q = q.masked_fill(expanded_mask == 0, float(-1e5))
        k = k.masked_fill(expanded_mask == 0, float(-1e5))
        # # Calculate the dot product
        score = q @ k.transpose(-2, -1) / math.sqrt(dimension)

        # # Apply mask
        # expanded_mask = mask.unsqueeze(1).expand_as(score)
        #
        # score = score.masked_fill(expanded_mask == 0, float(-1e10))  # float(-inf) will result in Nan
        score = self.dropout(self.softmax(score))  # dropout=0.3

        output = torch.matmul(score, v)

        return output[:, -1, :]  # return the last token's output

class CR(nn.Module):
    def __init__(self, conf, raw_graph, teacher_knowledge_feature, seq_matrix, l2_hyper):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"] + 1
        self.num_courses = conf["num_courses"] + 1
        self.num_exercises = conf["num_exercises"] + 1
        self.num_knowledge = conf["num_knowledge"] + 1

        self.max_seq_len = conf["max_seq_len"]
        # self.seq_timestamp = seq_timestamp
        self.group_num = conf['group_num']
        self.group_intra_factor = conf['group_intra_factor']
        self.group_inter_factor = conf['group_inter_factor']
        self.group_partition = conf['group_partition']
        # self.time_decay_matrix = self.cal_time_decay_matrix((self.num_users, self.max_seq_len, 1),
        #                                                     self.group_intra_factor,
        #                                                     self.group_inter_factor, self.group_partition)

        self.init_emb()
        self.teacher_knowledge_feature = teacher_knowledge_feature

        self.mlp = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size)

        assert isinstance(raw_graph, list)
        self.uc_graph, self.uew_graph, self.ck_graph, self.ek_graph, self.ce_graph, self.cu_graph, self.euw_graph = raw_graph
        self.uk_graph = self.uew_graph@self.ek_graph
        self.h_graph = sp.vstack((self.cu_graph, self.euw_graph))

        # self.ue_graph = to_tensor(self.ue_graph).to_dense().to(self.device)

        # generate the graph without any dropouts for testing
        self.get_exercises_level_graph_ori()
        self.get_exercises_level_graph()

        self.get_courses_level_graph_ori()
        self.get_courses_level_graph()

        self.get_knowledge_level_graph_ori()
        self.get_knowledge_level_graph()

        self.get_courses_agg_graph()
        self.get_courses_agg_graph_ori()

        self.get_exercises_agg_graph()
        self.get_exercises_agg_graph_ori()

        self.get_hyper_graph()
        self.get_hyper_graph_ori()

        self.seq_matrix = seq_matrix

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        # self.get_exercises_level_graph()


        self.init_md_dropouts()

        self.attention_layer = diy_attention(self.embedding_size)
        self.num_head = 4
        self.multi_head_attention_layer = torch.nn.MultiheadAttention(self.embedding_size, self.num_head, 0.3,
                                                                      batch_first=True)

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]
        self.l2_hyper = l2_hyper

    def init_md_dropouts(self):
        self.exercises_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.courses_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.courses_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.courses_feature = nn.Parameter(torch.FloatTensor(self.num_courses, self.embedding_size))
        nn.init.xavier_normal_(self.courses_feature)
        self.exercises_feature = nn.Parameter(torch.FloatTensor(self.num_exercises, self.embedding_size))
        nn.init.xavier_normal_(self.exercises_feature)

        self.double_feature = nn.Parameter(
            torch.FloatTensor(self.num_courses + self.num_exercises, self.embedding_size))
        nn.init.xavier_normal_(self.double_feature)

    def get_exercises_level_graph(self):
        ue_graph = self.uew_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        exercises_level_graph = sp.bmat([[sp.csr_matrix((ue_graph.shape[0], ue_graph.shape[0])), ue_graph],
                                         [ue_graph.T, sp.csr_matrix((ue_graph.shape[1], ue_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = exercises_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                exercises_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.exercises_level_graph = to_tensor(laplace_transform(exercises_level_graph)).to(device)

    def get_exercises_level_graph_ori(self):
        ue_graph = self.uew_graph
        device = self.device
        exercises_level_graph = sp.bmat([[sp.csr_matrix((ue_graph.shape[0], ue_graph.shape[0])), ue_graph],
                                         [ue_graph.T, sp.csr_matrix((ue_graph.shape[1], ue_graph.shape[1]))]])
        self.exercises_level_graph_ori = to_tensor(laplace_transform(exercises_level_graph)).to(device)

    def get_knowledge_level_graph(self):
        uk_graph = self.uk_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        knowledge_level_graph = sp.bmat([[sp.csr_matrix((uk_graph.shape[0], uk_graph.shape[0])), uk_graph],
                                       [uk_graph.T, sp.csr_matrix((uk_graph.shape[1], uk_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = knowledge_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                knowledge_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.knowledge_level_graph = to_tensor(laplace_transform(knowledge_level_graph)).to(device)

    def get_knowledge_level_graph_ori(self):
        uk_graph = self.uk_graph
        device = self.device

        knowledge_level_graph = sp.bmat([[sp.csr_matrix((uk_graph.shape[0], uk_graph.shape[0])), uk_graph],
                                         [uk_graph.T, sp.csr_matrix((uk_graph.shape[1], uk_graph.shape[1]))]])
        self.knowledge_level_graph_ori = to_tensor(laplace_transform(knowledge_level_graph)).to(device)

    def get_courses_level_graph(self):
        uc_graph = self.uc_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        courses_level_graph = sp.bmat([[sp.csr_matrix((uc_graph.shape[0], uc_graph.shape[0])), uc_graph],
                                       [uc_graph.T, sp.csr_matrix((uc_graph.shape[1], uc_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = courses_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                courses_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.courses_level_graph = to_tensor(laplace_transform(courses_level_graph)).to(device)

    def get_courses_level_graph_ori(self):
        uc_graph = self.uc_graph
        device = self.device
        courses_level_graph = sp.bmat([[sp.csr_matrix((uc_graph.shape[0], uc_graph.shape[0])), uc_graph],
                                       [uc_graph.T, sp.csr_matrix((uc_graph.shape[1], uc_graph.shape[1]))]])
        self.courses_level_graph_ori = to_tensor(laplace_transform(courses_level_graph)).to(device)
    def get_hyper_graph(self):
        hyper_graph = self.h_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        hyper_graph = sp.bmat([[sp.csr_matrix((hyper_graph.shape[0], hyper_graph.shape[0])), hyper_graph],
                                      [hyper_graph.T, sp.csr_matrix((hyper_graph.shape[1], hyper_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = hyper_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                hyper_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.hyper_graph = to_tensor(laplace_transform(hyper_graph)).to(device)

    def get_hyper_graph_ori(self):
        hyper_graph = self.h_graph
        device = self.device
        exercises_level_graph = sp.bmat([[sp.csr_matrix((hyper_graph.shape[0], hyper_graph.shape[0])), hyper_graph],
                                    [hyper_graph.T, sp.csr_matrix((hyper_graph.shape[1], hyper_graph.shape[1]))]])
        self.hyper_graph_ori = to_tensor(laplace_transform(exercises_level_graph)).to(device)

    def get_courses_agg_graph(self):
        ce_graph = self.ce_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.ce_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            ce_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        course_size = ce_graph.sum(axis=1) + 1e-8
        ce_graph = sp.diags(1/course_size.A.ravel()) @ ce_graph
        self.courses_agg_graph = to_tensor(ce_graph).to(device)


    def get_courses_agg_graph_ori(self):
        ce_graph = self.ce_graph
        device = self.device

        course_size = ce_graph.sum(axis=1) + 1e-8
        ce_graph = sp.diags(1 / course_size.A.ravel()) @ ce_graph
        self.courses_agg_graph_ori = to_tensor(ce_graph).to(device)

    def get_exercises_agg_graph(self):
        ek_graph = self.ek_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.ek_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            ek_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        exercise_size = ek_graph.sum(axis=1) + 1e-8
        ek_graph = sp.diags(1 / exercise_size.A.ravel()) @ ek_graph
        self.exercises_agg_graph = to_tensor(ek_graph).to(device)


    def get_exercises_agg_graph_ori(self):
        ek_graph = self.ek_graph
        device = self.device

        exercise_size = ek_graph.sum(axis=1) + 1e-8
        ek_graph = sp.diags(1 / exercise_size.A.ravel()) @ ek_graph
        self.exercises_agg_graph_ori = to_tensor(ek_graph).to(device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)  # 提取特征
            if self.conf["aug_type"] == "MD" and not test:  # !!! important
                features = mess_dropout(features)

            features = features / (i + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)  # 堆叠
        all_features = torch.sum(all_features, dim=1).squeeze(1)  # 求和

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def get_KL_course_rep(self, knowledge_feature, test):
        if test:
            course_feature = torch.matmul(self.courses_agg_graph_ori, knowledge_feature)
        else:
            course_feature = torch.matmul(self.courses_agg_graph, knowledge_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            course_feature = self.courses_agg_dropout(course_feature)

        return course_feature

    def get_KL_exercise_rep(self, knowledge_feature, test):
        if test:
            exercise_feature = torch.matmul(self.exercises_agg_graph_ori, knowledge_feature)
        else:
            exercise_feature = torch.matmul(self.exercises_agg_graph, knowledge_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            exercise_feature = self.exercises_agg_dropout(exercise_feature)

        return exercise_feature

    def propagate(self, test=False):
        #  =============================  exercise level propagation  =============================
        # if test:
        #     e_users_feature, exercises_feature = self.one_propagate(self.exercises_level_graph, self.users_feature,
        #                                                              self.exercises_feature,
        #                                                              self.exercises_level_dropout, test)
        # else:
        #     e_users_feature, exercises_feature = self.one_propagate(self.exercises_level_graph_ori, self.users_feature,
        #                                                              self.exercises_feature,
        #                                                              self.exercises_level_dropout, test)
        #
        #  # ============================= course level propagation =============================
        # if test:
        #     c_users_feature, c_courses_feature = self.one_propagate(self.courses_level_graph, self.users_feature,
        #                                                            self.courses_feature, self.courses_level_dropout,
        #                                                            test)
        # else:
        #     c_users_feature, c_courses_feature = self.one_propagate(self.courses_level_graph_ori, self.users_feature,
        #                                                            self.courses_feature, self.courses_level_dropout,
        #                                                            test)

        # if test:
        #    k_users_feature, knowledge_feature = self.one_propagate(self.knowledge_level_graph, self.users_feature,
        #                                                           self.knowledge_feature, self.exercises_level_dropout,
        #                                                           test)
        #else:
        #    k_users_feature, knowledge_feature = self.one_propagate(self.knowledge_level_graph_ori, self.users_feature,
        #                                                           self.knowledge_feature, self.exercises_level_dropout,
        #                                                           test)

        # knowledge_feature = self.relu(self.mlp(torch.cat((knowledge_feature, self.teacher_knowledge_feature), 1)))

        if test:
            double_features, s_users_feature = self.one_propagate(self.hyper_graph, self.double_feature,
                                                                      self.users_feature,
                                                                      self.courses_level_dropout,
                                                                      test)
        else:
            double_features, s_users_feature = self.one_propagate(self.hyper_graph_ori, self.double_feature,
                                                                      self.users_feature,
                                                                      self.courses_level_dropout,
                                                                      test)

        s_courses_feature, exercises_feature = torch.split(double_features, (
            self.courses_feature.shape[0], self.exercises_feature.shape[0]), 0)

        e_courses_feature = self.get_KL_course_rep(exercises_feature, test)

        courses_feature = self.mlp(torch.cat((s_courses_feature,  e_courses_feature), dim=1))

        seq_matrix_new = np.zeros([self.num_users, self.max_seq_len, self.embedding_size])
        for index in range(len(self.seq_matrix)):
            for inter in range(len(self.seq_matrix[index])):
                if self.seq_matrix[index][inter][0] != 0:
                    if self.seq_matrix[index][inter][1] == 0:
                        emb = exercises_feature[int(self.seq_matrix[index][inter][0])].cpu().detach().numpy()
                        seq_matrix_new[index][inter] = emb
                    # 如果seq_matrix最后一维的标识为1，则标识是课程
                    elif self.seq_matrix[index][inter][1] == 1:
                        emb = courses_feature[int(self.seq_matrix[index][inter][0])].cpu().detach().numpy()
                        seq_matrix_new[index][inter] = emb

        mask = (torch.tensor(seq_matrix_new) != 0).float()

        # time_decay_matrix = self.time_decay_matrix
        # expand_time_decay_matrix = time_decay_matrix.expand(seq_matrix_new.shape)
        seq_matrix_new = torch.tensor(seq_matrix_new, dtype=torch.float32).to(self.device)
        # 元素乘
        # seq_matrix_new = torch.mul(expand_time_decay_matrix, seq_matrix_new)

        seq_attn_emb = self.attention_layer(seq_matrix_new, mask.to(self.device))

        k_users_feature = self.teacher_knowledge_feature
        gate = self.sigmoid(self.w1(s_users_feature) + self.w2(k_users_feature))
        users_feature = self.mlp(torch.cat([gate * s_users_feature, (1 - gate) * k_users_feature], -1))
        users_feature = self.relu(users_feature)


        users_feature = users_feature + seq_attn_emb

        return users_feature, courses_feature, s_users_feature, k_users_feature

    # def cal_time_decay_matrix(self, _matrix_shape: tuple, group_num: int, group_intra_factor: float,
    #                           group_inter_factor: float, group_partition: list) -> torch.Tensor:
    def cal_time_decay_matrix(self, _matrix_shape: tuple, group_intra_factor: float,
                              group_inter_factor: float, group_partition: int) -> torch.Tensor:
        """
        计算时间衰退矩阵
        :param _matrix_shape: 矩阵维度
        :param group_intra_factor: 组内衰退因子
        :param group_inter_factor: 组间衰退因子
        :param group_partition: 划分组的间隔天数
        :return:
        """
        res = torch.ones(_matrix_shape, requires_grad=False, device=self.device)
        # # 组间
        # idx_group_start = 0
        # group_num = len(group_partition)
        # for idx_group in range(group_num):
        #     idx_group_end = idx_group_start + group_partition[idx_group]
        #     if idx_group == group_num - 1:
        #         idx_group_end = _matrix_shape[0]
        #     # 组内
        #     for idx_intra_group in range(idx_group_start, idx_group_end):
        #         res[idx_intra_group, :] = math.pow(group_intra_factor, idx_group_end - idx_intra_group - 1)
        #     res[idx_group_start: idx_group_end, :] = (res[idx_group_start: idx_group_end, :] *
        #                                               math.pow(group_inter_factor, group_num - idx_group))
        #     idx_group_start = idx_group_end
        # return res
        seq_timestamp = self.seq_timestamp
        # 从最后的天数开始，如果当前和最后的间隔不足 group_partition，则归为该组
        for uid in tqdm.tqdm(seq_timestamp.keys(), desc="计算时间衰退矩阵"):
            group_inter_number = group_intra_number = 0
            seq = seq_timestamp[uid]
            if len(seq) == 0:
                continue
            last_timestamp = seq[-1]
            l = len(seq)
            for i in range(l):
                oi = i
                # l-1 l-2 ... 2 1 0
                i = l - i - 1
                if (last_timestamp - seq[i]).days <= group_partition:
                    res[uid, -oi - 1] = (math.pow(group_inter_factor, group_inter_number) *
                                         math.pow(group_intra_factor, group_intra_number))
                    group_intra_number += 1
                else:
                    group_inter_number += 1
                    group_intra_number = 0
                    last_timestamp = seq[i]
                    res[uid, -oi - 1] = (math.pow(group_inter_factor, group_inter_number) *
                                         math.pow(group_intra_factor, group_intra_number))

        return res

    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp)  # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)  # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_loss(self, users_feature, courses_feature):

        # c_users_feature, k_users_feature = users_feature
        # c_courses_feature, k_courses_feature = courses_feature

        pred = torch.sum(users_feature * courses_feature, 2)
        bpr_loss = cal_bpr_loss(pred)
        # loss_l2 = torch.norm(bpr_loss, p=2)
        # loss = bpr_loss + loss_l2 * 0.3
        return bpr_loss

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            # self.get_exercises_level_graph()
            self.get_courses_level_graph()
            # self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, courses = batch
        users_feature, courses_feature, s_users_feature, k_users_feature = self.propagate()

        # users_embedding = [i[users].expand(-1, courses.shape[1], -1) for i in users_feature]
        # courses_embedding = [i[courses] for i in courses_feature]
        users_embedding = users_feature[users].expand(-1, courses.shape[1], -1)
        courses_embedding = courses_feature[courses]

        bpr_loss = self.cal_loss(users_embedding, courses_embedding)
        kd_loss = l2_regularizer(s_users_feature - k_users_feature, self.l2_hyper)

        loss = bpr_loss + kd_loss
        return loss, kd_loss

    def evaluate(self, propagate_result, users):

        users_feature, courses_feature, _, _ = propagate_result
        users_feature = users_feature[users]

        scores_rec = torch.mm(users_feature, courses_feature.t())

        return scores_rec
