import pickle

import numpy as np
import pandas as pd


def load_pickle_data(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)

    return data


def load_csv_data():
    X_1 = pd.read_csv('./data/PCP.csv', header=None)
    X_2 = pd.read_csv('./data/RFHC_GAC.csv', header=None)
    X = {0: X_1.values, 1: X_2.values}

    y = np.array([np.ones((int(X_1.shape[0] / 2), 1)), np.zeros((int(X_1.shape[0] / 2), 1))]).reshape(-1)
    data_num = X_1.shape[0]
    return X, y, data_num


def load_fasta_data(dataset, type, TrainDownSampleSize_pos, TrainDownSampleSize_neg, TestDownSampleSize_pos,
                    TestDownSampleSize_neg):
    if dataset == 1:
        pos_data_names = ['mm10_pos_seqs_100_sramp', 'hg19_pos_seqs_100_sramp']
        neg_data_names = ['mm10_neg_seqs_100_sramp', 'hg19_neg_seqs_100_sramp']
    elif dataset == 2:
        pos_data_names = ['mm10_pos_seqs_100_sramp', 'hg19_pos_seqs_100_sramp']
        neg_data_names = ['mm10_neg_seqs_100_sramp', 'hg19_neg_seqs_100_sramp']
    elif dataset == 3:
        if type == 'train':
            pos_data_names = ['hg19_pos_seqs_1000_100',
                              'panTro4_pos_seqs_1000_100', 'rheMac8_pos_seqs_1000_100',
                              'mm10_pos_seqs_1000_100', 'rn5_pos_seqs_1000_100',
                              'susScr3_pos_seqs_1000_100',
                              'danRer10_pos_seqs_1000_100']
            neg_data_names = ['hg19_neg_seqs_1000_100',
                              'panTro4_neg_seqs_1000_100', 'rheMac8_neg_seqs_1000_100',
                              'mm10_neg_seqs_1000_100', 'rn5_neg_seqs_1000_100',
                              'susScr3_neg_seqs_1000_100',
                              'danRer10_neg_seqs_1000_100']
        elif type == 'test':
            pos_data_names = ['hg19_pos_seqs_1000_100_self_processed', 'panTro4_pos_seqs_1000_100_self_processed',
                              'rheMac8_pos_seqs_1000_100_self_processed',
                              'mm10_pos_seqs_1000_100_self_processed', 'rn5_pos_seqs_1000_100_self_processed',
                              'susScr3_pos_seqs_1000_100_self_processed',
                              'danRer10_pos_seqs_1000_100_self_processed']
            neg_data_names = ['hg19_neg_seqs_1000_100_self_processed', 'panTro4_neg_seqs_1000_100_self_processed',
                              'rheMac8_neg_seqs_1000_100_self_processed',
                              'mm10_neg_seqs_1000_100_self_processed', 'rn5_neg_seqs_1000_100_self_processed',
                              'susScr3_neg_seqs_1000_100_self_processed',
                              'danRer10_neg_seqs_1000_100_self_processed']
    elif dataset == 4:
        pos_data_names = ['hs_pos', 'mm_pos']
        neg_data_names = ['hs_neg', 'mm_neg']

    pos_seq = []
    neg_seq = []
    len_pos_seq = []
    len_neg_seq = []

    for pos_data_name, neg_data_name in zip(pos_data_names, neg_data_names):

        if type == 'train':
            pos_seq_dir = "./data/data%d/positive_samples/" % (
                dataset) + type + '/' + pos_data_name + TrainDownSampleSize_pos
            neg_seq_dir = "./data/data%d/negative_samples/" % (
                dataset) + type + '/' + neg_data_name + TrainDownSampleSize_neg
        elif type == 'test':
            pos_seq_dir = "./data/data%d/positive_samples/" % (
                dataset) + type + '/' + pos_data_name + TestDownSampleSize_pos
            neg_seq_dir = "./data/data%d/negative_samples/" % (
                dataset) + type + '/' + neg_data_name + TestDownSampleSize_neg

        pos_seq_tmp = []
        neg_seq_tmp = []

        fh = open(pos_seq_dir)
        for line in fh:
            pos_seq_tmp.append(line.replace('/n', ''))
        fh.close()
        pos_seq.append(pos_seq_tmp)
        len_pos_seq.append(len(pos_seq_tmp))

        fh = open(neg_seq_dir)
        for line in fh:
            neg_seq_tmp.append(line.replace('/n', ''))
        fh.close()
        neg_seq.append(neg_seq_tmp)
        len_neg_seq.append(len(neg_seq_tmp))

    return pos_seq, neg_seq, len_pos_seq, len_neg_seq


def load_m4A_data():
    path = 'D:/acw/学习研究/多视角学习/code/code_acw_MvLapSRC_HSIC/data/m4A/'
    names = ['A1978', 'C1554', 'D1769', 'E388', 'Gpick569', 'Gsub905']
    types = ['Negative', 'Positive']
    pos_seq = []
    neg_seq = []
    len_pos_seq = []
    len_neg_seq = []

    for name in names:
        pos_seq_tmp = []
        neg_seq_tmp = []

        pos_seq_dir = path + types[1] + '_' + name
        if name == 'Gsub905':
            neg_seq_dir = path + types[0] + '_' + 'Gsub906'
        else:
            neg_seq_dir = path + types[0] + '_' + name

        fh = open(pos_seq_dir)
        for line in fh:
            pos_seq_tmp.append(line.replace('\n',''))
        fh.close()
        pos_seq.append(pos_seq_tmp)
        len_pos_seq.append(len(pos_seq_tmp))

        fh = open(neg_seq_dir)
        for line in fh:
            neg_seq_tmp.append(line.replace('\n', ''))
        fh.close()
        neg_seq.append(neg_seq_tmp)
        len_neg_seq.append(len(neg_seq_tmp))

    return pos_seq, neg_seq, len_pos_seq, len_neg_seq

def load_4mC_data():
    path = './data/4mC/'
    names = ['A1978', 'C1554', 'D1769', 'E388', 'Gpick569', 'Gsub905']
    types = ['Negative', 'Positive']
    pos_seq = []
    neg_seq = []
    len_pos_seq = []
    len_neg_seq = []

    for name in names:
        pos_seq_tmp = []
        neg_seq_tmp = []

        pos_seq_dir = path + types[1] + '_' + name
        if name == 'Gsub905':
            neg_seq_dir = path + types[0] + '_' + 'Gsub906'
        else:
            neg_seq_dir = path + types[0] + '_' + name

        fh = open(pos_seq_dir)
        for line in fh:
            pos_seq_tmp.append(line.replace('\n',''))
        fh.close()
        pos_seq.append(pos_seq_tmp)
        len_pos_seq.append(len(pos_seq_tmp))

        fh = open(neg_seq_dir)
        for line in fh:
            neg_seq_tmp.append(line.replace('\n', ''))
        fh.close()
        neg_seq.append(neg_seq_tmp)
        len_neg_seq.append(len(neg_seq_tmp))

    return pos_seq, neg_seq, len_pos_seq, len_neg_seq
