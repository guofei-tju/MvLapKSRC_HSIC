from sklearn import metrics

from config import *


class MvKernelLapSRC():

    def __init__(self):
        return None

    def kernel_RBF_v(self, x, y, gamma):

        z = x - y
        r2 = z.dot(z).astype(config.DataFormat)
        return np.exp(-r2 * gamma, dtype=config.DataFormat)

    def kernel_RBF(self, X, Y, gamma):
        X = X.T
        Y = Y.T
        r2 = np.repeat(np.sum(X * X, 1), len(Y)).reshape(-1, len(Y)) + \
             np.repeat(np.sum(Y * Y, 1), len(X)).reshape(-1, len(X)).T - \
             2 * np.dot(X, Y.T)
        return np.exp(-r2 * gamma)

    def kernel_function(self, X, Y, gamma, type):
        if type == 'rbf':
            return metrics.pairwise.rbf_kernel(X, Y, gamma)
        elif type == 'linear':
            return metrics.pairwise.linear_kernel(X, Y)
        elif type == 'poly2':
            return metrics.pairwise.polynomial_kernel(X, Y, degree=2)

    def train(self, train_x_list, train_y, test_x_list, lambda_r=4, mu=0.1, theta=1, gamma=1, maxIteration=3, kf='rbf', V=1):
        '''
        Multi-view Kernel Laplacian Regularized Sparse Representation-based Classifier (Multi-view Kernel LapRSRC)
        tju cs, bioinformatics.
        :parameter
            %train_x_list: the multi-view object of matrices of training set, (n-by-d-by-v)
            %train_y: the labels of training set, (n-by-1)
            %test_x_list:  the multi-view list of matrices of test set, (m-by-d-by-v)
            %lambda_r: Regularized item parameter for L2,1
            %mu: Regularized item parameter for Laplacian
            %theta: Regularized item parameter for HISC
            %gamma: bandwidth of RBF kernel
            %maxIteration: The maximum number of iterations.
            %kf :kernel function
            %Alpha: Sparse representation coefficient matrix of A
            %distance_s: the Reconstruction error for each class
            %predict_y: the Predicted label for test samples
        '''
        np.random.seed(12345678)
        loss_list=[]
        # change the dtype of data
        Trainlabels = train_y
        uniqlabels = np.unique(train_y)
        c = np.size(uniqlabels)
        all_view_distance_s = []
        all_view_scores = []
        all_Alpha_array = []
        n_test = np.size(test_x_list[0], 0)
        n_train = np.size(train_x_list[0], 0)

        rand_split_index = np.arange(n_test)
        # np.random.shuffle(rand_split_index)
        n_test_index_split = np.array_split(rand_split_index, config.TEST.split_size)
        distance_s = []
        for test_split_index in n_test_index_split:
            test_split = test_x_list[0][test_split_index]
            n_test_split = np.size(test_split, 0)
            split_distances = np.zeros((V, n_test_split, c), dtype=config.DataFormat)
            Alpha_array = np.random.rand(V, n_train, n_test_split)
            for o in np.arange(maxIteration):
                loss = 0
                KRSL = 0
                for v in range(V):
                    K_xx = self.kernel_function(train_x_list[v], train_x_list[v], gamma[v], kf)
                    K_yy = self.kernel_function(test_x_list[v][test_split_index], test_x_list[v][test_split_index],
                                                gamma[v], kf)
                    K_xy = self.kernel_function(train_x_list[v], test_x_list[v][test_split_index], gamma[v], kf)
                    # K_xx_L_M = self.kernel_function(train_x_list[v], train_x_list[v], gamma[v], 'rbf')
                    e = np.ones((n_train,1))
                    H = np.diag(e.reshape(-1)) - (e * e.T / n_train)
                    # Kw = np.zeros([n_train, n_train], dtype=config.DataFormat)
                    # KwA = np.zeros([n_train, n_test_split], dtype=config.DataFormat)

                    A = Alpha_array[v, :, :]

                    L_M = self.Lap_M_computing(K_xx)

                    G = np.linalg.pinv(np.diag(np.sqrt((A ** 2).sum(axis=1))))

                    K_hisc_w = np.zeros((n_train, n_train))
                    for w in np.arange(V):
                        if w != v:
                            K_hisc_w += (H.dot(Alpha_array[w, :, :]).dot((Alpha_array[w, :, :]).T).dot(H))

                    loss += np.trace(A.T.dot(K_xx).dot(A) - 2 * A.T.dot(K_xy) + K_yy) / 2 + \
                            lambda_r[v] * np.trace(A.T.dot(G).dot(A)) + mu[v] * np.trace(
                        A.T.dot(L_M).dot(A)) / 2 +  theta[v] * np.trace(A.T.dot(K_hisc_w).dot(A)) / 2

                    A = np.linalg.inv(K_xx + lambda_r[v] * G + mu[v] * L_M + theta[v] * K_hisc_w).dot(K_xy)

                    # A = np.linalg.inv(K_xx + lambda_r[v] * G + mu[v] * L_M + theta * Kw).dot(K_xy + KwA)
                    Alpha_array[v, :, :] = A

                print("iteration: ", o, " ,the loss is: ", loss)
                loss_list.append(loss)

            eps = 1e-3
            Alpha_array[np.abs(Alpha_array) <= eps] = 0

            # time_start = time.time()

            for v in range(V):
                K_xx = self.kernel_function(train_x_list[v], train_x_list[v], gamma[v], kf)
                K_yy = self.kernel_function(test_x_list[v][test_split_index], test_x_list[v][test_split_index],
                                            gamma[v], kf)
                K_xy = self.kernel_function(train_x_list[v], test_x_list[v][test_split_index], gamma[v], kf)
                for i in range(c):
                    train_x_c_index = np.array(np.where(Trainlabels == uniqlabels[i])).reshape(-1)
                    K_i = K_xx[train_x_c_index][:, train_x_c_index]
                    A_i = Alpha_array[v, train_x_c_index]
                    dis = np.diagonal(A_i.T.dot(K_i).dot(A_i) - 2 * A_i.T.dot(K_xy[train_x_c_index]) + K_yy)
                    split_distances[v, :, i] = np.sqrt(dis)

                # for j in np.arange(n_test_split):
                #     for i in np.arange(c):
                #         dis = 0
                #         train_x_c_index = np.array(np.where(Trainlabels == uniqlabels[i])).reshape(-1)
                #         K_i = K_xx[train_x_c_index][:, train_x_c_index]
                #         A_i = Alpha_array[v, train_x_c_index, j]
                #         dis += A_i.T.dot(K_i).dot(A_i) - 2 * A_i.T.dot(K_xy[train_x_c_index, j]) + K_yy[
                #             j, j]
                #
                #         split_distances[v, j, i] = dis

            # time_end = time.time()
            # print('totally cost', time_end - time_start)
            # calculate the mean view split distance of test set
            split_distances = np.mean(split_distances, 0)

            distance_s.append(split_distances)
            all_Alpha_array.append(Alpha_array)

        all_Alpha_array = np.concatenate(all_Alpha_array, 2)
        distance_s = np.concatenate(distance_s)
        predict_score = 1 - (distance_s[:, 1]) / (distance_s[:, 0] + distance_s[:, 1])

        indices = np.argmin(distance_s, 1)
        predict_y = uniqlabels[indices]

        return predict_y, predict_score, distance_s, all_Alpha_array

    def Lap_M_computing(self, adj_matrix):
        D = np.diag(adj_matrix.sum(axis=1))
        D_1_2 = np.diag(adj_matrix.sum(axis=1) ** -0.5)

        return D_1_2.dot(D - adj_matrix).dot(D_1_2)

    def cal_matrix_L2(self, X):
        # cal L2 norm of every col vec of a matrix
        n = np.size(X, 1)
        w = np.zeros((n, 1))
        for i in range(n):
            w[i] = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        return w
