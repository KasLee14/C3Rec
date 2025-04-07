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
    def __init__(self, d_model, max_seq_length, time_decay_matrix: torch.Tensor):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.time_decay_matrix = time_decay_matrix.expand((*time_decay_matrix.shape[:-1], d_model))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return torch.mul(x, self.time_decay_matrix)


class diy_attention(nn.Module):
    def __init__(self, d_model, dropout=0.3, time_decay_matrix: torch.Tensor = None) -> None:
        super(diy_attention, self).__init__()

        self.d_model = d_model
        self.dropout = Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.PositionalEncoding = PositionalEncoding(d_model, 50, time_decay_matrix=time_decay_matrix)

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

class CD(nn.Module):
    def __init__(self, conf, raw_graph):
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
        self.batch_size_train = conf["batch_size_train"]
        self.prednet_len1, self.prednet_len2 = 512, 256
        # self.max_seq_len = conf["max_seq_len"]
        #
        # self.seq_timestamp = seq_timestamp
        self.group_num = conf['group_num']
        self.group_intra_factor = conf['group_intra_factor']
        self.group_inter_factor = conf['group_inter_factor']
        self.group_partition = conf['group_partition']
        # self.time_decay_matrix = self.cal_time_decay_matrix((self.num_users, self.max_seq_len, 1),
        #                                                     self.group_intra_factor,
        #                                                     self.group_inter_factor, self.group_partition)

        self.init_emb()

        self.mlp = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        assert isinstance(raw_graph, list)
        self.ue_graph, self.ek_graph = raw_graph
        self.uk_graph = self.ue_graph@self.ek_graph

        # self.ue_graph = to_tensor(self.ue_graph).to_dense().to(self.device)

        # generate the graph without any dropouts for testing
        # self.get_exercises_level_graph_ori()
        self.get_exercises_level_graph_ori()
        self.get_exercises_level_graph()

        self.get_knowledge_level_graph_ori()
        self.get_knowledge_level_graph()

        self.get_exercises_agg_graph()
        self.get_exercises_agg_graph_ori()
        # self.seq_matrix = seq_matrix

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        # self.get_exercises_level_graph()


        self.init_md_dropouts()

        # self.attention_layer = diy_attention(self.embedding_size, time_decay_matrix=self.time_decay_matrix)
        self.num_head = 4
        self.multi_head_attention_layer = torch.nn.MultiheadAttention(self.embedding_size, self.num_head, 0.3,
                                                                      batch_first=True)

        self.prednet_full1 = nn.Linear(self.batch_size_train, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

    def init_md_dropouts(self):
        # self.exercises_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.exercises_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.exercises_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.exercises_feature = nn.Parameter(torch.FloatTensor(self.num_exercises, self.embedding_size))
        nn.init.xavier_normal_(self.exercises_feature)
        self.knowledge_feature = nn.Parameter(torch.FloatTensor(self.num_knowledge, self.embedding_size))
        nn.init.xavier_normal_(self.knowledge_feature)
        # self.exercises_feature = nn.Parameter(torch.FloatTensor(self.num_exercises, self.embedding_size))
        # nn.init.xavier_normal_(self.exercises_feature)

    def get_exercises_level_graph(self):
        ue_graph = self.ue_graph
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
        ue_graph = self.ue_graph
        device = self.device
        exercises_level_graph = sp.bmat([[sp.csr_matrix((ue_graph.shape[0], ue_graph.shape[0])), ue_graph],
                                         [ue_graph.T, sp.csr_matrix((ue_graph.shape[1], ue_graph.shape[1]))]])
        self.exercises_level_graph_ori = to_tensor(laplace_transform(exercises_level_graph)).to(device)

    def get_knowledge_level_graph(self):
        ek_graph = self.ek_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        knowledge_level_graph = sp.bmat([[sp.csr_matrix(( ek_graph.shape[0],  ek_graph.shape[0])),  ek_graph],
                                       [ ek_graph.T, sp.csr_matrix(( ek_graph.shape[1],  ek_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = knowledge_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                knowledge_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.knowledge_level_graph = to_tensor(laplace_transform(knowledge_level_graph)).to(device)

    def get_knowledge_level_graph_ori(self):
        ek_graph = self.ek_graph
        device = self.device

        knowledge_level_graph = sp.bmat([[sp.csr_matrix((ek_graph.shape[0], ek_graph.shape[0])), ek_graph],
                                         [ek_graph.T, sp.csr_matrix((ek_graph.shape[1], ek_graph.shape[1]))]])
        self.knowledge_level_graph_ori = to_tensor(laplace_transform(knowledge_level_graph)).to(device)


    def get_exercises_agg_graph(self):
        ek_graph = self.ek_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.ek_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            ek_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        exercise_size = ek_graph.sum(axis=1) + 1e-8
        ek_graph = sp.diags(1/exercise_size.A.ravel()) @ ek_graph
        self.exercises_agg_graph = to_tensor(ek_graph).to(device)


    def get_exercises_agg_graph_ori(self):
        ek_graph = self.ek_graph
        device = self.device

        exercise_size = ek_graph.sum(axis=1) + 1e-8
        ek_graph = sp.diags(1 / exercise_size.A.ravel()) @ ek_graph
        self.exercises_agg_graph = to_tensor(ek_graph).to(device)


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

    def get_KL_exercise_rep(self, knowledge_feature, test):
        if test:
            knowledge_feature = torch.matmul(self.exercises_agg_graph_ori, knowledge_feature)
        else:
            knowledge_feature = torch.matmul(self.exercises_agg_graph, knowledge_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            knowledge_feature = self.exercises_agg_dropout(knowledge_feature)

        return knowledge_feature

    def propagate(self, test=False):
        #  =============================  exercise level propagation  =============================
        # if test:
        #     EL_users_feature, exercises_feature = self.one_propagate(self.exercises_level_graph, self.users_feature,
        #                                                              self.exercises_feature,
        #                                                              self.exercises_level_dropout, test)
        # else:
        #     EL_users_feature, exercises_feature = self.one_propagate(self.exercises_level_graph_ori, self.users_feature,
        #                                                              self.exercises_feature,
        #                                                              self.exercises_level_dropout, test)

         # ============================= course level propagation =============================
        if test:
            users_feature, e_exercises_feature = self.one_propagate(self.exercises_level_graph, self.users_feature,
                                                                   self.exercises_feature, self.exercises_level_dropout,
                                                                   test)
        else:
            users_feature, e_exercises_feature = self.one_propagate(self.exercises_level_graph_ori, self.users_feature,
                                                                   self.exercises_feature, self.exercises_level_dropout,
                                                                   test)

        if test:
            k_exercises_feature, knowledge_feature = self.one_propagate(self.knowledge_level_graph, self.exercises_feature,
                                                                   self.knowledge_feature, self.exercises_level_dropout,
                                                                   test)
        else:
            k_exercises_feature, knowledge_feature = self.one_propagate(self.knowledge_level_graph_ori, self.exercises_feature,
                                                                   self.knowledge_feature, self.exercises_level_dropout,
                                                                   test)

        knowledge_feature = self.get_KL_exercise_rep(knowledge_feature, test)
        users_feature = self.sigmoid(users_feature)
        exercises_feature = torch.cat((e_exercises_feature, k_exercises_feature), dim=1)
        exercises_feature = self.sigmoid(self.mlp(exercises_feature))
        # knowledge_feature = self.sigmoid(knowledge_feature)

        return users_feature, exercises_feature, knowledge_feature


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

        # pred = torch.sum(c_users_feature * c_courses_feature, 2) + torch.sum(k_users_feature * k_courses_feature, 2)
        pred = torch.sum(users_feature * courses_feature, 2)
        bpr_loss = cal_bpr_loss(pred)


        return bpr_loss

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            # self.get_exercises_level_graph()
            self.get_exercises_level_graph()
            # self.get_bundle_agg_graph()

        users, exercises, labels = batch
        users_feature, exercises_feature, knowledge_feature = self.propagate()


        users_embedding = users_feature[users]
        exercises_embedding = exercises_feature[exercises]
        knowledge_embedding = knowledge_feature[exercises]


        input_x = (users_embedding - knowledge_embedding) * exercises_embedding
        input_x = input_x[:, -1, :]
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        loss = nn.MSELoss()
        loss = loss(output[:, -1], labels[:, -1])

        # bpr_loss = self.cal_loss(users_embedding, courses_embedding)

        return loss, users_feature

    def evaluate(self, propagate_result, users):

        users_feature, courses_feature = propagate_result
        users_feature = users_feature[users]

        scores_rec = torch.mm(users_feature, courses_feature.t())

        return scores_rec
