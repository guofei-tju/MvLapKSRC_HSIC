from scipy.io import loadmat

from MvKernelLapSRC import MvKernelLapSRC
from construct_features import *
from load_data import load_4mC_data
from measurement_tools import performance


def normalization(data, dim=2, V=1):
    data_v = []
    if dim == 2:
        for i in range(np.shape(data)[1]):
            _range = np.max(data[:, i]) - np.min(data[:, i])
            data_v.append((data[:, i] - np.min(data[:, i])) / _range)
        data_v = np.vstack(data_v).T
    elif dim == 3:
        for d in range(V):
            data_d = []
            for i in range(np.shape(data[d])[1]):
                _range = np.max(data[d][:, i]) - np.min(data[d][:, i])

                data_d.append((data[d][:, i] - np.min(data[d][:, i])) / _range)
            data_d = np.vstack(data_d).T
            data_v.append(data_d)
        # data_v = np.array(data_v)

    return data_v


def MvPS3merNP(all_positive_seq, all_negative_seq, train_sample, test_sample, Z_score, V, interval):
    RNA_code = 'ACGT'

    tes_final_value = []
    tra_final_value = []

    len_seq = len(all_positive_seq[0][0])
    code_values = make_kmer_list(interval, RNA_code)
    code_len = len(code_values)
    all_positive_seq_value = loadmat('./data/4mC/features/PSTNP/Z_score/Z_score_positive.mat')
    all_positive_seq_value = all_positive_seq_value['Z_score_positive']
    all_negative_seq_value = loadmat('./data/4mC/features/PSTNP/Z_score/Z_score_negative.mat')
    all_negative_seq_value = all_negative_seq_value['Z_score_negative']

    for z in Z_score:

        # calculate Z matrix
        positive_seq_value = all_positive_seq_value[z]
        negative_seq_value = all_negative_seq_value[z]

        #  training features
        tra_positive_seq = all_positive_seq[train_sample]
        tra_negative_seq = all_negative_seq[train_sample]

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

        #  testing features
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

    X_train = np.array(tra_final_value)
    X_test = np.array(tes_final_value)

    return X_train, X_test


np.random.seed(1)
names = ['A1978', 'C1554', 'D1769', 'E388', 'Gpick569', 'Gsub905']

# parameter setting
# Numbers 0,1,2,3,4,5 represent species A.thaliana,C.elegans,D.melanogaster,E.coli,G.pickeringi, and G.subterraneus respectively


Z_score = [0, 1]  # PSTNP profiles
tra_spe = 0  # training species
tes_spe = 2  # testing species
V = len(Z_score)

runs = [1]
maxIterations = [5]
type = 'rbf'

alphabets_lambda = [[0.001, 0.001]]
alphabets_mu = [[0.04, 0.04]]
alphabets_gamma = [[16, 5]]
alphabets_thetas = [[1, 2]]
alphabets = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

positive_seq, negative_seq, len_positive_seq, len_negative_seq = load_4mC_data()
# make train and test features
train_x, test_x = MvPS3merNP(positive_seq, negative_seq, tra_spe, tes_spe, Z_score, V=V, interval=3)
train_y = np.concatenate((np.ones(len_positive_seq[tra_spe]), np.zeros(len_negative_seq[tra_spe])))
test_y = np.concatenate((np.ones(len_positive_seq[tes_spe]), np.zeros(len_negative_seq[tes_spe])))

for run in runs:
    parameters = []
    results = []
    for maxIteration in maxIterations:
        for lambda_r in alphabets_lambda:  # itertools.product(alphabets, repeat=V):
            for mu in alphabets_mu:  # itertools.product(alphabets, repeat=V):
                for theta in alphabets_thetas:  # itertools.product(alphabets, repeat=V):
                    for gamma in alphabets_gamma:  # itertools.product(alphabets, repeat=V):
                        results_y = []

                        model = MvKernelLapSRC()
                        predict_y, predict_score, distance_s, ALPHA = model.train(train_x, train_y, test_x, lambda_r,
                                                                                  mu, theta, gamma, maxIteration,
                                                                                  type, V)

                        precision, recall, ACC, SN, SP, MCC, GM, TP, TN, FP, FN = performance(test_y, predict_y)

                        print('ACC: ', ACC)
                        print('sn: ', SN)
                        print('sp: ', SP)
                        print('MCC: ', MCC)
                        # save parameters
                        parameters.append([lambda_r, mu, theta, gamma, maxIteration, ACC, SN, SP, MCC])
                        parameter_header = ['lambda_r', 'mu', 'theta', 'gamma', 'maxIteration', 'acc', 'sn', 'sp',
                                            'mcc']
                        pdparameters = pd.DataFrame(parameters, columns=parameter_header)

                        writer = pd.ExcelWriter('./results/%s_view_grid_search_train%s_test[%s]_Zscore%s.xlsx' % (
                            V, names[tra_spe], names[tes_spe], Z_score))
                        pdparameters.to_excel(writer)
                        writer.save()

                        results_y.append([predict_y.tolist(), test_y.tolist()])
                        pdresults_y = pd.DataFrame(results_y)
                        pdresults_y.to_csv('./results/%s_view_results_train%s_test[%s]_Zscore%s.txt' % (
                        V, names[tra_spe], names[tes_spe], Z_score), sep='\n')
