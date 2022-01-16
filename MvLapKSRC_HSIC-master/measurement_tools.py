import math

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def performance(labelArr, predictArr):
    # labelArr[i] is actual value,predictArr[i] is predict value

    TP = 0.;
    TN = 0.;
    FP = 0.;
    FN = 0.;
    up_label = max(labelArr)
    down_label = min(labelArr)
    for i in range(len(labelArr)):
        if labelArr[i] == up_label and predictArr[i] == up_label:
            TP += 1.
        if labelArr[i] == up_label and predictArr[i] == down_label:
            FN += 1.
        if labelArr[i] == down_label and predictArr[i] == up_label:
            FP += 1.
        if labelArr[i] == down_label and predictArr[i] == down_label:
            TN += 1.
    if (TP + FN) == 0:
        SN = 0
    else:
        SN = TP / (TP + FN)  # Sensitivity = TP/P  and P = TP + FN
    if (FP + TN) == 0:
        SP = 0
    else:
        SP = TN / (FP + TN)  # Specificity = TN/N  and N = TN + FP
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (TP + FN) == 0 or (FP + TN) == 0 or (TP + FP) == 0 or (TP + FN) == 0:
        MCC = 0
    else:
        MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FN) * (FP + TN) * (TP + FP) * (TP + FN))
    GM = math.sqrt(recall * SP)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # AUC, AUPR = auc_aupr(labelArr, predictArr)
    return precision, recall, ACC, SN, SP, MCC, GM, TP, TN, FP, FN


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    # mcc_list = (TP*TN - FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    # mcc = mcc_list[max_index]
    tp = TP[max_index]
    tn = TN[max_index]
    fp = FP[max_index]
    fn = FN[max_index]
    return aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision, tp, tn, fp, fn


def auc_aupr(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    AUPR = auc(recall, precision)
    AUC = auc(fpr, tpr)

    return AUC, AUPR
