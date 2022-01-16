import itertools

import numpy as np
import pandas as pd


def F_score(v, y_label):
    x_0 = 0
    x_1 = 0
    v_pos = v[y_label > 0]
    v_neg = v[y_label <= 0]

    v_ave = np.mean(v)
    v_pos_ave = np.mean(v_pos)
    v_neg_ave = np.mean(v_neg)

    len_pos = len(v_pos)
    len_neg = len(v_neg)
    for i in range(len_pos):
        x_0 += (v_pos[i] - v_pos_ave) ** 2
    for j in range(len_neg):
        x_1 += (v_neg[i] - v_neg_ave) ** 2

    f_score = ((v_pos_ave - v_ave) ** 2 + (v_neg_ave - v_ave) ** 2) / (
                (1 / (len_pos - 1)) * x_0 + (1 / (len_neg - 1)) * x_1)

    return f_score


def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def kmer(data_seq, k):
    # calculate the k-mer feature of a seq
    RNA_code = 'ACGT'
    code_values = make_kmer_list(3, RNA_code)
    count = np.zeros((len(data_seq), len(code_values)))
    for i, line_value in enumerate(data_seq.values):  # for every samples
        for j, code_value in enumerate(line_value[0]):  # for every position
            if j <= len(line_value[0]) - k + 1:
                for p, c_value in enumerate(code_values):
                    if c_value == line_value[0][j:j + k]:
                        count[i][p] += 1
    count /= len(code_values) - k + 1
    return count


def MvPS3merNP(all_positive_seq, all_negative_seq, train_samples, test_sample, interval):
    RNA_code = 'ACGT'

    all_final_seq_value_tra = []
    all_final_seq_value_tes = []

    for train_sample in train_samples:

        # calculate Z matrix
        positive_seq = all_positive_seq[train_sample]
        negative_seq = all_negative_seq[train_sample]

        len_seq = len(positive_seq[0])
        positive_df = pd.DataFrame(positive_seq)
        positive_x_train = positive_df.iloc[:, :]

        negative_df = pd.DataFrame(negative_seq)
        negative_x_train = negative_df.iloc[:, :]

        code_values = make_kmer_list(interval, RNA_code)
        code_len = len(code_values)
        positive_seq_value = [[0 for jj in range(len_seq - interval + 1)] for ii in range(code_len)]
        negative_seq_value = [[0 for jj in range(len_seq - interval + 1)] for ii in range(code_len)]

        for i, line_value in enumerate(positive_x_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            positive_seq_value[p][j] += 1
        positive_seq_value = np.matrix(positive_seq_value) * 1.0 / (len(positive_seq))
        for i, line_value in enumerate(negative_x_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            negative_seq_value[p][j] += 1
        negative_seq_value = np.matrix(negative_seq_value) * 1.0 / (len(negative_seq))

        tes_final_value = []
        tra_final_value = []

        #  training features
        for train_sample_x in train_samples:

            tra_positive_seq = all_positive_seq[train_sample_x]
            tra_negative_seq = all_negative_seq[train_sample_x]

            tra_positive_df = pd.DataFrame(tra_positive_seq)
            tra_negative_df = pd.DataFrame(tra_negative_seq)
            tra_positive_train = tra_positive_df.iloc[:, :]
            tra_negative_train = tra_negative_df.iloc[:, :]
            tra_positive_negative_train = pd.concat([tra_positive_train, tra_negative_train], axis=0)
            tra_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
                                   range(len(tra_positive_negative_train))]

            for i, line_value in enumerate(tra_positive_negative_train.values):
                for j, code_value in enumerate(line_value[0]):
                    if j <= len(line_value[0]) - interval + 1:
                        for p, c_value in enumerate(code_values):
                            if c_value == line_value[0][j:j + interval]:
                                tra_final_seq_value[i][j] = positive_seq_value[p, j] - negative_seq_value[p, j]

            tra_final_value.append(tra_final_seq_value)

        tes_positive_seq = all_positive_seq[test_sample]
        tes_negative_seq = all_negative_seq[test_sample]

        tes_positive_df = pd.DataFrame(tes_positive_seq)
        tes_negative_df = pd.DataFrame(tes_negative_seq)
        tes_positive_train = tes_positive_df.iloc[:, :]
        tes_negative_train = tes_negative_df.iloc[:, :]
        tes_positive_negative_train = pd.concat([tes_positive_train, tes_negative_train], axis=0)
        tes_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
                               range(len(tes_positive_negative_train))]

        for i, line_value in enumerate(tes_positive_negative_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            tes_final_seq_value[i][j] = positive_seq_value[p, j] - negative_seq_value[p, j]

        tes_final_value.append(tes_final_seq_value)

        all_final_seq_value_tra.append(np.concatenate(tra_final_value))
        all_final_seq_value_tes.append(np.concatenate(tes_final_value))

    X_train = np.array(all_final_seq_value_tra)
    X_test = np.array(all_final_seq_value_tes)

    return X_train, X_test


def MvPS3merNP_KL(all_positive_seq, all_negative_seq, train_samples, test_sample, interval):
    RNA_code = 'ACGT'

    all_final_seq_value_tra = []
    all_final_seq_value_tes = []

    for train_sample in train_samples:

        # calculate Z matrix
        positive_seq = all_positive_seq[train_sample]
        negative_seq = all_negative_seq[train_sample]

        len_seq = len(positive_seq[0])
        positive_df = pd.DataFrame(positive_seq)
        positive_x_train = positive_df.iloc[:, :]

        negative_df = pd.DataFrame(negative_seq)
        negative_x_train = negative_df.iloc[:, :]

        code_values = make_kmer_list(interval, RNA_code)
        code_len = len(code_values)
        positive_seq_value = [[0 for jj in range(len_seq - interval + 1)] for ii in range(code_len)]
        negative_seq_value = [[0 for jj in range(len_seq - interval + 1)] for ii in range(code_len)]

        for i, line_value in enumerate(positive_x_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            positive_seq_value[p][j] += 1
        positive_seq_value = np.matrix(positive_seq_value) * 1.0 / (len(positive_seq))
        for i, line_value in enumerate(negative_x_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            negative_seq_value[p][j] += 1
        negative_seq_value = np.matrix(negative_seq_value) * 1.0 / (len(negative_seq))

        positive_seq_value[positive_seq_value <= 0] = 1e-09
        positive_seq_value_log = np.log(positive_seq_value)
        # positive_seq_value_log[np.isinf(positive_seq_value_log)] = -10
        negative_seq_value[negative_seq_value <= 0] = 1e-09
        negative_seq_value_log = np.log(negative_seq_value)
        # negative_seq_value_log[np.isinf(negative_seq_value_log)] = -10

        Z = np.multiply(positive_seq_value, (positive_seq_value_log - negative_seq_value_log))
        tes_final_value = []
        tra_final_value = []

        #  training features
        for train_sample_x in train_samples:

            tra_positive_seq = all_positive_seq[train_sample_x]
            tra_negative_seq = all_negative_seq[train_sample_x]

            tra_positive_df = pd.DataFrame(tra_positive_seq)
            tra_negative_df = pd.DataFrame(tra_negative_seq)
            tra_positive_train = tra_positive_df.iloc[:, :]
            tra_negative_train = tra_negative_df.iloc[:, :]
            tra_positive_negative_train = pd.concat([tra_positive_train, tra_negative_train], axis=0)
            tra_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
                                   range(len(tra_positive_negative_train))]

            for i, line_value in enumerate(tra_positive_negative_train.values):
                for j, code_value in enumerate(line_value[0]):
                    if j <= len(line_value[0]) - interval + 1:
                        for p, c_value in enumerate(code_values):
                            if c_value == line_value[0][j:j + interval]:
                                tra_final_seq_value[i][j] = Z[p, j]

            tra_final_value.append(tra_final_seq_value)

        tes_positive_seq = all_positive_seq[test_sample]
        tes_negative_seq = all_negative_seq[test_sample]

        tes_positive_df = pd.DataFrame(tes_positive_seq)
        tes_negative_df = pd.DataFrame(tes_negative_seq)
        tes_positive_train = tes_positive_df.iloc[:, :]
        tes_negative_train = tes_negative_df.iloc[:, :]
        tes_positive_negative_train = pd.concat([tes_positive_train, tes_negative_train], axis=0)
        tes_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
                               range(len(tes_positive_negative_train))]

        for i, line_value in enumerate(tes_positive_negative_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            tes_final_seq_value[i][j] = Z[p, j]

        tes_final_value.append(tes_final_seq_value)

        all_final_seq_value_tra.append(np.concatenate(tra_final_value))
        all_final_seq_value_tes.append(np.concatenate(tes_final_value))

    X_train = np.array(all_final_seq_value_tra)
    X_test = np.array(all_final_seq_value_tes)

    return X_train, X_test


def MvPS3merNP_JS(all_positive_seq, all_negative_seq, train_samples, test_sample, interval):
    RNA_code = 'ACGT'

    all_final_seq_value_tra = []
    all_final_seq_value_tes = []

    for train_sample in train_samples:

        # calculate Z matrix
        positive_seq = all_positive_seq[train_sample]
        negative_seq = all_negative_seq[train_sample]

        len_seq = len(positive_seq[0])
        positive_df = pd.DataFrame(positive_seq)
        positive_x_train = positive_df.iloc[:, :]

        negative_df = pd.DataFrame(negative_seq)
        negative_x_train = negative_df.iloc[:, :]

        code_values = make_kmer_list(interval, RNA_code)
        code_len = len(code_values)
        positive_seq_value = [[0 for jj in range(len_seq - interval + 1)] for ii in range(code_len)]
        negative_seq_value = [[0 for jj in range(len_seq - interval + 1)] for ii in range(code_len)]

        for i, line_value in enumerate(positive_x_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            positive_seq_value[p][j] += 1
        positive_seq_value = np.matrix(positive_seq_value) * 1.0 / (len(positive_seq))
        for i, line_value in enumerate(negative_x_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            negative_seq_value[p][j] += 1
        negative_seq_value = np.matrix(negative_seq_value) * 1.0 / (len(negative_seq))

        positive_seq_value[positive_seq_value <= 0] = 1e-09
        positive_seq_value_log = np.log(positive_seq_value)
        # positive_seq_value_log[np.isinf(positive_seq_value_log)] = -10
        negative_seq_value[negative_seq_value <= 0] = 1e-09
        negative_seq_value_log = np.log(negative_seq_value)
        # negative_seq_value_log[np.isinf(negative_seq_value_log)] = -10

        seq_value_log = np.log((positive_seq_value + negative_seq_value) / 2)

        Z = 1 / 2 * np.multiply(positive_seq_value, (positive_seq_value_log - seq_value_log)) + 1 / 2 * np.multiply(
            negative_seq_value, (negative_seq_value_log - seq_value_log))

        tes_final_value = []
        tra_final_value = []

        #  training features
        for train_sample_x in train_samples:

            tra_positive_seq = all_positive_seq[train_sample_x]
            tra_negative_seq = all_negative_seq[train_sample_x]

            tra_positive_df = pd.DataFrame(tra_positive_seq)
            tra_negative_df = pd.DataFrame(tra_negative_seq)
            tra_positive_train = tra_positive_df.iloc[:, :]
            tra_negative_train = tra_negative_df.iloc[:, :]
            tra_positive_negative_train = pd.concat([tra_positive_train, tra_negative_train], axis=0)
            tra_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
                                   range(len(tra_positive_negative_train))]

            for i, line_value in enumerate(tra_positive_negative_train.values):
                for j, code_value in enumerate(line_value[0]):
                    if j <= len(line_value[0]) - interval + 1:
                        for p, c_value in enumerate(code_values):
                            if c_value == line_value[0][j:j + interval]:
                                tra_final_seq_value[i][j] = Z[p, j]

            tra_final_value.append(tra_final_seq_value)

        tes_positive_seq = all_positive_seq[test_sample]
        tes_negative_seq = all_negative_seq[test_sample]

        tes_positive_df = pd.DataFrame(tes_positive_seq)
        tes_negative_df = pd.DataFrame(tes_negative_seq)
        tes_positive_train = tes_positive_df.iloc[:, :]
        tes_negative_train = tes_negative_df.iloc[:, :]
        tes_positive_negative_train = pd.concat([tes_positive_train, tes_negative_train], axis=0)
        tes_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
                               range(len(tes_positive_negative_train))]

        for i, line_value in enumerate(tes_positive_negative_train.values):
            for j, code_value in enumerate(line_value[0]):
                if j <= len(line_value[0]) - interval + 1:
                    for p, c_value in enumerate(code_values):
                        if c_value == line_value[0][j:j + interval]:
                            tes_final_seq_value[i][j] = Z[p, j]

        tes_final_value.append(tes_final_seq_value)

        all_final_seq_value_tra.append(np.concatenate(tra_final_value))
        all_final_seq_value_tes.append(np.concatenate(tes_final_value))

    X_train = np.array(all_final_seq_value_tra)
    X_test = np.array(all_final_seq_value_tes)

    return X_train, X_test


def PRO12(all_positive_seq, all_negative_seq, train_sample):
    RNA_code = 'ACGT'
    Z = {'AA': [0.85, 0.85, -0.22, -0.96, -0.73, -0.62, -0.09, 0.68, -1, 1, 0.36, 0.23],
         'AC': [-1, -1, -0.6, -0.45, -0.27, 0.37, -0.64, 0.37, 0.05, 0.56, -0.48, 0.50],
         'AG': [-0.56, -0.56, -1.00, 0.45, -0.27, -0.18, -0.36, 0.37, -0.12, -0.10, -0.39, 0.04],
         'AT': [-0.01, -0.01, 1.00, -1.00, -1.00, -0.48, -1.00, 1.00, -0.31, -0.90, -1.00, 1.00],
         'CA': [1.00, 1.00, 0.13, 1.00, -0.27, -0.65, -0.09, 0.16, 0.75, -0.14, 0.88, -0.77],
         'CC': [-0.87, -0.87, -0.25, 0.81, 1.00, 0.15, 1.00, -0.47, 1.00, -0.75, -0.15, -0.35],
         'CG': [-0.14, -0.14, -0.89, 0.80, 0.18, -0.10, 1.45, -1.00, 0.64, -0.45, 0.60, -1.00],
         'CT': [-0.56, -0.56, -1.00, -0.29, -0.27, -0.18, -0.36, 0.37, -0.12, -1.00, -0.39, 0.04],
         'GA': [0.87, 0.87, 0.43, 1.24, -0.27, -0.30, -0.36, 0.37, -0.02, 0.87, 0.65, 0.04],
         'GC': [0.32, 0.32, 0.24, 1.17, 0.18, 1.00, 1.00, -0.47, 0.44, -0.54, 0.01, 0.27],
         'GG': [-0.87, -0.87, -0.25, 0.63, 1.00, 0.15, 1.00, -0.47, 1.00, -0.14, -0.15, -0.35],
         'GT': [-1.00, -1.00, -0.60, -0.29, -0.27, 0.37, -0.64, 0.37, 0.05, -0.90, -0.48, 0.50],
         'TA': [0.32, 0.32, -0.84, 2.37, -1.00, -1.00, -0.45, 1.00, 0.29, -0.87, 1.00, -0.31],
         'TC': [0.87, 0.87, 0.43, 0.24, -0.27, -0.30, -0.36, 0.37, -0.02, -0.45, 0.65, 0.04],
         'TG': [1.00, 1.00, 0.13, 2.02, -0.27, -0.65, -0.09, 0.16, 0.75, 0.56, 0.88, -0.77],
         'TT': [0.85, 0.85, -0.22, -1.00, -0.73, -0.62, -0.09, 0.68, -1.00, -0.77, 0.36, 0.23]}
    code_values = make_kmer_list(2, RNA_code)

    positive_seq = all_positive_seq[train_sample]
    negative_seq = all_negative_seq[train_sample]

    len_seq = len(positive_seq[0])
    positive_df = pd.DataFrame(positive_seq)
    positive_x_train = positive_df.iloc[:, :]

    negative_df = pd.DataFrame(negative_seq)
    negative_x_train = negative_df.iloc[:, :]
    positive_negative_train = pd.concat([positive_x_train, negative_x_train], axis=0)

    tra_final_seq_value = [[0 for ii in range(12 * (len_seq - 1))] for jj in
                           range(len(positive_negative_train))]

    for i, line_value in enumerate(positive_negative_train.values):  # for every samples
        for j, code_value in enumerate(line_value[0]):  # for every position
            if j <= len(line_value[0]) - 2 + 1:
                for p, c_value in enumerate(code_values):
                    if c_value == line_value[0][j:j + 2]:
                        tra_final_seq_value[i][(12 * j):(12 * (j + 1))] = np.array(Z[c_value])

    tra_final_seq_value = np.array(tra_final_seq_value)
    return tra_final_seq_value


def ratio(all_positive_seq, all_negative_seq, train_sample):
    # calculate the GC content, GC skew,AT skew ,AT/GC ratio
    positive_seq = all_positive_seq[train_sample]
    negative_seq = all_negative_seq[train_sample]

    positive_df = pd.DataFrame(positive_seq)
    positive_x_train = positive_df.iloc[:, :]

    negative_df = pd.DataFrame(negative_seq)
    negative_x_train = negative_df.iloc[:, :]
    positive_negative_train = pd.concat([positive_x_train, negative_x_train], axis=0)

    tra_final_seq_value = np.zeros((len(positive_negative_train), 4))

    for i, line_value in enumerate(positive_negative_train.values):  # for every samples

        A_count = line_value[0].count('A')
        C_count = line_value[0].count('C')
        G_count = line_value[0].count('G')
        T_count = line_value[0].count('T')
        tra_final_seq_value[i][0] = (C_count + G_count) / len(positive_negative_train)
        tra_final_seq_value[i][1] = (C_count - G_count) / (C_count + G_count)
        tra_final_seq_value[i][2] = (A_count - T_count) / (A_count + T_count)
        tra_final_seq_value[i][3] = (A_count + T_count) / (C_count + G_count)

    return tra_final_seq_value


def ChemicalProperty(all_positive_seq, all_negative_seq, train_sample):
    RNA_code = 'ACGT'
    X = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1]}

    positive_seq = all_positive_seq[train_sample]
    negative_seq = all_negative_seq[train_sample]
    len_seq = len(positive_seq[0])

    positive_df = pd.DataFrame(positive_seq)
    positive_x_train = positive_df.iloc[:, :]

    negative_df = pd.DataFrame(negative_seq)
    negative_x_train = negative_df.iloc[:, :]
    positive_negative_train = pd.concat([positive_x_train, negative_x_train], axis=0)
    code_values = make_kmer_list(1, RNA_code)
    tra_final_seq_value = [[0 for ii in range(4 * (len_seq))] for jj in
                           range(len(positive_negative_train))]
    for i, line_value in enumerate(positive_negative_train.values):  # for every samples
        for j, code_value in enumerate(line_value[0]):
            if j <= len(line_value[0]) - 1 + 1:
                for p, c_value in enumerate(code_values):
                    if c_value == line_value[0][j:j + 1]:
                        X_c_value = X[c_value] + [(line_value[0][:j + 1].count(c_value)) / (j + 1)]
                        tra_final_seq_value[i][(4 * j):(4 * (j + 1))] = X_c_value

    tra_final_seq_value = np.array(tra_final_seq_value)
    return tra_final_seq_value


def PseEiip(all_positive_seq, all_negative_seq, train_sample):
    RNA_code = 'ACGT'
    ea = 0.126
    et = 0.1335
    eg = 0.0806
    ec = 0.134
    eACGT = {'A': ea, 'C': et, 'G': eg, 'T': ec}

    code_values = make_kmer_list(3, RNA_code)
    emer = np.zeros((1, len(code_values)))

    positive_seq = all_positive_seq[train_sample]
    negative_seq = all_negative_seq[train_sample]

    len_seq = len(positive_seq[0])
    positive_df = pd.DataFrame(positive_seq)
    positive_x_train = positive_df.iloc[:, :]

    negative_df = pd.DataFrame(negative_seq)
    negative_x_train = negative_df.iloc[:, :]
    positive_negative_train = pd.concat([positive_x_train, negative_x_train], axis=0)

    for i, code_value in enumerate(code_values):
        emer[0][i] = eACGT[code_value[0]] + eACGT[code_value[1]] + eACGT[code_value[2]]

    EMER = np.ones((len(positive_negative_train), 1)).dot(emer)
    F = kmer(positive_negative_train, 3)
    A = F * EMER
    return A
