#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>' * 10 + string + '>' * 10)
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice) / X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice) / X.shape[1])
    print('Matrix density', len(nonzero_row_indice) / (X.shape[0] * X.shape[1]))


class TrainDataset(Dataset):
    def __init__(self, conf, u_c_pairs, u_c_graph, num_users, num_courses, neg_sample=1):
        self.conf = conf
        self.u_c_pairs = u_c_pairs
        self.u_c_graph = u_c_graph

        self.num_users = num_users
        self.num_courses = num_courses
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        conf = self.conf
        user_c, pos_courses = self.u_c_pairs[index]
        all_courses = [pos_courses]

        while True:
            i = np.random.randint(self.num_courses)
            if self.u_c_graph[user_c, i] == 0 and not i in all_courses:
                all_courses.append(i)
                if len(all_courses) == self.neg_sample + 1:
                    break

        return torch.LongTensor([user_c]), torch.LongTensor(all_courses)

    def __len__(self):
        return len(self.u_c_pairs)


class TestDataset(Dataset):
    def __init__(self, u_c_pairs, u_c_graph, u_c_graph_train, num_users, num_courses):
        self.u_c_pairs = u_c_pairs
        self.u_c_graph = u_c_graph
        self.train_mask_u_c = u_c_graph_train

        self.num_users = num_users
        self.num_courses = num_courses

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.courses = torch.arange(num_courses, dtype=torch.long)

    def __getitem__(self, index):
        u_c_grd = torch.from_numpy(self.u_c_graph[index].toarray()).squeeze()
        u_c_mask = torch.from_numpy(self.train_mask_u_c[index].toarray()).squeeze()

        return index, u_c_grd, u_c_mask

    def __len__(self):
        return self.u_c_graph.shape[0]


class CR_Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        self.ub_weights = conf['ub_weights']
        batch_size_train = conf['batch_size_train']

        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_courses, self.num_exercises, self.num_knowledge = self.get_data_size()
        print(self.num_users, self.num_courses, self.num_exercises, self.num_knowledge)
        self.sequence = defaultdict(list)
        self.max_seq_len = conf['max_seq_len']

        u_e_pairs, u_e_w_graph, u_e_graph, e_u_w_graph = self.get_ue()

        u_c_pairs_train, u_c_graph_train, c_u_graph_train = self.get_uc("train")
        u_c_pairs_val, u_c_graph_val, c_u_graph_train  = self.get_uc("tune")
        u_c_pairs_test, u_c_graph_test, c_u_graph_train  = self.get_uc("test")
        _, c_k_graph = self.get_ck()
        _, e_k_graph = self.get_ek()
        _, c_e_graph = self.get_ce()

        self.train_data = TrainDataset(conf, u_c_pairs_train, u_c_graph_train, self.num_users, self.num_courses,
                                       conf["neg_num"])
        self.val_data = TestDataset(u_c_pairs_val, u_c_graph_val, u_c_graph_train, self.num_users, self.num_courses)
        self.test_data = TestDataset(u_c_pairs_test, u_c_graph_test, u_c_graph_train, self.num_users, self.num_courses)

        self.graphs = [u_c_graph_train, u_e_w_graph, c_k_graph, e_k_graph, c_e_graph, c_u_graph_train, e_u_w_graph]
        self.seq_matrix, self.seq_timestamp = self.get_seq()

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size_train, shuffle=False, num_workers=10,
                                       drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size_test, shuffle=True, num_workers=20)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size_test, shuffle=True, num_workers=20)

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')]

    def get_ue(self):
        with open(os.path.join(self.path, self.name, 'user_exercises_train.txt'), 'r') as f:
            u_e_t_pairs = [tuple(int(i) if idx < 3 else i for idx, i in enumerate(s.strip().split('\t'))) for s in
                           f.readlines()]

        u_e_pairs = [(user, item, correct) for user, item, correct, _ in u_e_t_pairs]
        indice_w = np.array(u_e_pairs, dtype=np.int32)
        values_w = np.zeros(len(u_e_pairs), dtype=np.float32)
        indice = np.array(u_e_pairs, dtype=np.int32)
        values = np.zeros(len(u_e_pairs), dtype=np.float32)
        for i in range(len(u_e_pairs)):
            if indice_w[i][2] != 0:
                values_w[i] = 1
            else:
                values_w[i] = self.ub_weights

        for i in range(len(u_e_pairs)):
            if indice_w[i][2] != 0:
                values[i] = 1
            else:
                values[i] = 0

        for pair in u_e_t_pairs:
            user, item, correct, time_str = pair[0], pair[1], pair[2], pair[3]
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            self.sequence[user].append((item, time, 0))

        u_e_w_graph = sp.coo_matrix(
            (values_w, (indice_w[:, 0], indice_w[:, 1])), shape=(self.num_users + 1, self.num_exercises + 1)).tocsr()

        u_e_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users + 1, self.num_exercises + 1)).tocsr()

        e_u_w_graph = sp.coo_matrix(
            (values_w, (indice_w[:, 1], indice_w[:, 0])), shape=(self.num_exercises + 1, self.num_users + 1)).tocsr()

        print_statistics(u_e_w_graph, 'U-E-W statistics')
        print_statistics(u_e_graph, 'U-E statistics')

        return u_e_pairs, u_e_w_graph, u_e_graph, e_u_w_graph

    def get_uc(self, task):
        with open(os.path.join(self.path, self.name, 'user_courses_{}.txt'.format(task)), 'r') as f:
            u_c_t_pairs = [tuple(int(i) if idx < 2 else i for idx, i in enumerate(s.strip().split('\t'))) for s in
                           f.readlines()]
        u_c_pairs = [(user, bundle) for user, bundle, _ in u_c_t_pairs]
        indice = np.array(u_c_pairs, dtype=np.int32)
        values = np.ones(len(u_c_pairs), dtype=np.float32)

        for pair in u_c_t_pairs:
            user, bundle, time_str = pair[0], pair[1], pair[2]
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            self.sequence[user].append((bundle, time, 1))

        u_c_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users + 1, self.num_courses + 1)).tocsr()

        c_u_graph = sp.coo_matrix(
            (values, (indice[:, 1], indice[:, 0])), shape=(self.num_courses + 1, self.num_users + 1)).tocsr()

        print_statistics(u_c_graph, "U-C statistics in %s" % (task))

        return u_c_pairs, u_c_graph, c_u_graph

    def get_ce(self):
        with open(os.path.join(self.path, self.name, 'course_exercise.txt'), 'r') as f:
            c_e_pairs = [tuple(int(i) for idx, i in enumerate(s.strip().split('\t'))) for s in
                         f.readlines()]
        indice = np.array(c_e_pairs, dtype=np.int32)
        values = np.ones(len(c_e_pairs), dtype=np.float32)
        c_e_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_courses + 1, self.num_exercises + 1)).tocsr()

        print_statistics(c_e_graph, "C-E statistics")

        return c_e_pairs, c_e_graph

    def get_ck(self):
        with open(os.path.join(self.path, self.name, 'course_topic.txt'), 'r') as f:
            c_k_pairs = [tuple(int(i) for idx, i in enumerate(s.strip().split('\t'))) for s in
                         f.readlines()]
        indice = np.array(c_k_pairs, dtype=np.int32)
        values = np.ones(len(c_k_pairs), dtype=np.float32)
        c_k_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_courses + 1, self.num_knowledge + 1)).tocsr()

        print_statistics(c_k_graph, "C-K statistics")

        return c_k_pairs, c_k_graph

    def get_ek(self):
        with open(os.path.join(self.path, self.name, 'exercise_topic.txt'), 'r') as f:
            e_k_pairs = [tuple(int(i) for idx, i in enumerate(s.strip().split('\t'))) for s in
                         f.readlines()]
        indice = np.array(e_k_pairs, dtype=np.int32)
        values = np.ones(len(e_k_pairs), dtype=np.float32)
        e_k_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_exercises + 1, self.num_knowledge + 1)).tocsr()

        print_statistics(e_k_graph, "E-K statistics")

        return e_k_pairs, e_k_graph

    def get_seq(self):
        # with open(os.path.join(self.path, self.name, 'user_course_challenge.txt'), 'r') as f:
        #     u_c_e_t_pairs = [tuple(int(i) if idx < 3 else i for idx, i in enumerate(s.strip().split('\t'))) for s in
        #                      f.readlines()]
        # u_c_e_pairs = [(user, course, challenge) for user, course, challenge, _, _ in u_c_e_t_pairs]

        # with open(os.path.join(self.path, self.name, 'user_challenge.txt'), 'r') as f:
        #     u_e_t_pairs = [tuple(int(i) if idx < 3 else i for idx, i in enumerate(s.strip().split('\t'))) for s in f.readlines()]
        # u_e_pairs = [(user, exercise) for user, exercise, _, _ in u_e_t_pairs]
        # indice = np.array(u_e_pairs, dtype=np.int32)
        # values = np.ones(len(u_e_pairs), dtype=np.float32)
        # u_e_graph = sp.coo_matrix(
        #     (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_challenge)).tocsr()
        max_sequence_length = self.max_seq_len

        sequence_timestamps = defaultdict(list)
        for user in self.sequence:
            self.sequence[user].sort(key=lambda x: x[1])
            tmp = [(item, t) for item, time, t in self.sequence[user]]
            sequence_timestamps[user] = [time for item, time, t in self.sequence[user]][:max_sequence_length]
            self.sequence[user] = tmp


        # initialize matrix with 0
        sequence_matrix = np.zeros([self.num_users + 1, max_sequence_length, 2])

        for user_id, sequence in self.sequence.items():
            sequence = np.array(sequence)
            if len(sequence) > max_sequence_length:
                sequence = np.array(sequence[:max_sequence_length])
            sequence_matrix[user_id, -len(sequence):] = sequence
        return sequence_matrix, sequence_timestamps

