from scipy.io import savemat
from tqdm import tqdm

from construct_features import *
from load_data import load_m4A_data

def write_to_csv(encodings, file):
    with open(file, 'w') as f:
        for line in encodings[0:]:
            line = line[0:]
            f.write('%s' % line[0])
            for i in range(1, len(line)):
                f.write(',%s' % line[i])
            f.write('\n')


def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def MvPS1merNP(all_positive_seq, all_negative_seq, test_sample, V, interval):
    RNA_code = 'ACGT'

    all_final_seq_value = []
    all_y_final_seq_value = []

    for v in range(V):

        # calculate Z matrix
        positive_seq = all_positive_seq[test_sample]
        negative_seq = all_negative_seq[test_sample]

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

        # v_final_seq_value = []
        # #  training features
        #
        # tes_positive_seq = all_positive_seq[test_sample]
        # tes_negative_seq = all_negative_seq[test_sample]
        #
        # t_positive_df = pd.DataFrame(tes_positive_seq)
        # t_negative_df = pd.DataFrame(tes_negative_seq)
        # w_positive_y_train = t_positive_df.iloc[:, :]
        # w_negative_y_train = t_negative_df.iloc[:, :]
        # w_positive_negative_y_train = pd.concat([w_positive_y_train, w_negative_y_train], axis=0)
        # y_final_seq_value = [[0 for ii in range(len_seq - interval + 1)] for jj in
        #                      range(len(w_positive_negative_y_train))]
        #
        # for i, line_value in enumerate(w_positive_negative_y_train.values):
        #     for j, code_value in enumerate(line_value[0]):
        #         if j <= len(line_value[0]) - interval + 1:
        #             for p, c_value in enumerate(code_values):
        #                 if c_value == line_value[0][j:j + interval]:
        #                     y_final_seq_value[i][j] = positive_seq_value[p, j] - negative_seq_value[p, j]
        #
        # v_final_seq_value.append(y_final_seq_value)
        #
        # all_y_final_seq_value.append(np.concatenate(v_final_seq_value))

    # X_train = np.array(all_y_final_seq_value)

    return positive_seq_value, negative_seq_value


K = 3
positive_seq, negative_seq, len_positive_seq, len_negative_seq = load_m4A_data()
names = ['A1978', 'C1554', 'D1769', 'E388', 'Gpick569', 'Gsub905']
for spe in tqdm(range(len(len_positive_seq))):
    Z_score_positive,Z_score_negative = MvPS1merNP(positive_seq, negative_seq, spe, V=1, interval=K)

    file_pos = 'D:/acw/学习研究/多视角学习/data/4mC/features/PSTNP/Z_score/pos_%s.csv' % (names[spe])
    file_neg = 'D:/acw/学习研究/多视角学习/data/4mC/features/PSTNP/Z_score/neg_%s.csv' % (names[spe])
    write_to_csv(Z_score_positive, file_pos)

    # savemat('%s_PSTNP.mat' % (names[spe]), {'%s' % (names[spe]): X_MvPS3merNP})
