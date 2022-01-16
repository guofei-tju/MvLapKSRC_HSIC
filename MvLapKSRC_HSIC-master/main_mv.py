import pandas as pd
import numpy as np
from tqdm import tqdm

from MvKernelLapSRC import MvKernelLapSRC
#from config import *
from measurement_tools import performance


def normalization(data, dim=2, V=1):
    data_v = []
    if dim == 2:
        for i in range(np.shape(data)[1]):
            _range = np.max(data[:, i]) - np.min(data[:, i])
            if _range == 0:
                continue
            data_v.append((data[:, i] - np.min(data[:, i])) / _range)
        data_v = np.vstack(data_v).T
    elif dim == 3:
        for d in range(V):
            data_d = []
            for i in range(np.shape(data[d])[1]):
                _range = np.max(data[d][:, i]) - np.min(data[d][:, i])
                if _range == 0:
                    continue
                data_d.append((data[d][:, i] - np.min(data[d][:, i])) / _range)
            data_d = np.vstack(data_d).T
            data_v.append(data_d)
        # data_v = np.array(data_v)

    return data_v


X = []
np.random.seed(1)
spe_names = ['A3956', 'C3108', 'D3538', 'E776', 'Gpick1138', 'Gsub1811']

# parameter setting
# Numbers 0,1,2,3,4,5 represent species A.thaliana,C.elegans,D.melanogaster,E.coli,G.pickeringi, and G.subterraneus respectively

V = 2
type = 'rbf'
spe = spe_names[3]
feature = 'NCP'

feature_path = './data/4mC/features/'
label_path = './data/4mC/labels/'

X_NCP = pd.read_csv(feature_path + '/%s/%s_%s.csv' % (feature, feature, spe), header=None)
X_PRO12 = pd.read_csv(feature_path + '/PRO12/PRO12_%s.csv' % (spe), header=None)
X_PSTNP = pd.read_csv(feature_path + '/PSTNP/PSTNP_%s.csv' % (spe), header=None)

X_NCP = X_NCP.values[:, 1:]
X_PRO12 = X_PRO12.values
X_PSTNP = X_PSTNP.values

X_chem = np.concatenate((X_PRO12, X_NCP), 1)

X.append(X_PSTNP)
X.append(X_chem)
y = pd.read_csv(label_path + 'y_%s.csv' % (spe), header=None)
y = y.values.reshape(-1)

X = normalization(X, dim=3, V=V)
len_seq = np.size(y)

runs = [1]
maxIterations = [5]
n_folds = 10
alphabets_lambda = [[0.0001, 0.0001]]
alphabets_mu = [[0.02, 0.001]]
alphabets_gamma = [[0.02, 0.015]]
alphabets_thetas = [[0.2, 0.001]]
lambda_krsl = 0
gamma_krsl = 0
# V = np.size(len_positive_seq)

for run in runs:
    parameters = []
    results = []
    for maxIteration in maxIterations:
        for lambda_r in alphabets_lambda:
            for mu in alphabets_mu:
                for theta in alphabets_thetas:
                    for gamma in alphabets_gamma:
                        mean_acc = []
                        fold_acc = []
                        mean_sn = []
                        fold_sn = []
                        mean_sp = []
                        fold_sp = []
                        mean_mcc = []
                        fold_mcc = []
                        X_Y_dis = []
                        X_Y_test_label = []

                        results_y = []

                        rand_pos_split_index = np.arange(len_seq)
                        np.random.shuffle(rand_pos_split_index)
                        dataset_pos_index = np.array_split(rand_pos_split_index, n_folds)

                        for i in tqdm(range(n_folds)):
                            train_x = []
                            test_x = []

                            train_index_tmp = dataset_pos_index.copy()
                            test_index = train_index_tmp.pop(i)
                            train_index = np.hstack(train_index_tmp)

                            for v in range(V):
                                train_x_v = X[v][train_index]
                                test_x_v = X[v][test_index]

                                train_x.append(train_x_v)
                                test_x.append(test_x_v)

                            train_y = y[train_index].reshape(-1)
                            test_y = y[test_index].reshape(-1)

                            model = MvKernelLapSRC()
                            predict_y, predict_score, distance_s, ALPHA = model.train(train_x, train_y, test_x,
                                                                                      lambda_r, mu, theta, gamma,
                                                                                      maxIteration, type, V)
                            precision, recall, ACC, SN, SP, MCC, GM, TP, TN, FP, FN = performance(test_y, predict_y)

                            fold_acc.append(ACC)
                            fold_sn.append(SN)
                            fold_sp.append(SP)
                            fold_mcc.append(MCC)

                        mean_acc = np.mean(fold_acc)
                        mean_sn = np.mean(fold_sn)
                        mean_sp = np.mean(fold_sp)
                        mean_mcc = np.mean(fold_mcc)

                        print('ACC: ', mean_acc)
                        print('sn: ', mean_sn)
                        print('sp: ', mean_sp)
                        print('MCC: ', mean_mcc)
                        print('...')

                        parameters.append(
                            [lambda_r, mu, theta, gamma, maxIteration, mean_acc, mean_sn, mean_sp, mean_mcc])
                        parameter_header = ['lambda_r', 'mu', 'theta', 'gamma', 'maxIteration', 'mean_acc',
                                            'mean_sn', ' mean_sp', 'mean_mcc']
                        pdparameters = pd.DataFrame(parameters, columns=parameter_header)

                        writer = pd.ExcelWriter(
                            './results/%s_view_grid_search_%s_%s&PRO12_PSTNP.xlsx' % (V, spe, feature))
                        pdparameters.to_excel(writer)
                        writer.save()
