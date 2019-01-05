#!usr/bin/python3
"""
This is an implementation of several evaluation methods as an external library to be used to evaluate machine learning algorithms
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np
from NB_Model import NaiveBayes
from LR_Model import LogisticRegression

def precision(y_true, y_pred):
    """
    calculates the precision score of this prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    if sum(y_pred) == 0:
        return 1.0
    # number of true positive / number of predicted positive
    return  sum([i == j for i,j in zip(y_true,y_pred) if j==True])/float(sum(y_pred))

def recall(y_true, y_pred):
    """
    calculates the recall score of this prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    if sum(y_true) == 0:
        return 1.0
    # number of true positive / number of factual true labels
    return sum([i == j for i, j in zip(y_true, y_pred) if i == True]) / float(sum(y_true))

def accuracy(y_true, y_pred):
    """
    calculate the accuracy of prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    assert len(y_true) == len(y_pred)
    count = 0
    for i, _ in enumerate(y_true):
        if y_true[i] == y_pred[i]:
            count += 1
    return count / float(len(y_true))

def specificity(y_true, y_pred):
    """
    calculate the specificity of this prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """

    if y_true.count(0) == 0:
        return 1.0

    return sum([i == j for i,j in zip(y_true,y_pred) if i==False]) / float(y_true.count(0))

def k_fold_cv(algo, algo_param, data, k):
    """
    perform k fold cross validation on the model
    ----------
    model : the model produced by machine learning algorithm
          a model to be cross validated
    data : array-like
          the entire dataset
    k : int
          the parameter in cross validation determing how many fold we're doing
    """
    if not ('nbayes' in algo or 'logreg' in algo):
        raise Exception ("INPUT_ALGORITHM_UNAVAILABLE")

    data_split = np.array_split(data, k)
    acc = []
    prcisn = []
    rcll = []

    # [confidence, true_label] with shape of (n, 2)
    conf_label_pair = np.array([[], []]).T

    np.random.seed(12345)
    np.random.shuffle(data)

    for i in range(0, k):
        if algo == 'nbayes':
            model = NaiveBayes(*algo_param)
        else:
            model = LogisticRegression(*algo_param)

        train_data = np.delete(data_split, (i), axis=0)
        train_data = np.concatenate(train_data)
        test_data = data_split[i]
        train_samples = train_data[:, 1:-1]
        train_targets = train_data[:, -1]
        test_samples = test_data[:, 1:-1]
        test_targets = [bool(x) for x in test_data[:, -1]]

        model.fit(train_samples, train_targets)

        result = np.array([model.predict(test_samples[j, :]) for j in range(test_samples.shape[0])])
        pred = result[:, 0]
        # [confidence, true_labels]
        new_pair = np.array([result[:, 1], test_targets]).T
        conf_label_pair = np.append(conf_label_pair, new_pair, axis=0)
        acc.append(accuracy(test_targets, pred))
        prcisn.append(precision(test_targets, pred))
        rcll.append(recall(test_targets, pred))

    avg_vals = [sum(acc) / float(k) , sum(prcisn) / float(k) , sum(rcll) / float(k)]
    std = [np.std(acc) , np.std(prcisn) , np.std(rcll)]

    # sort by confidence
    conf_label_pair = conf_label_pair[conf_label_pair[:, 0].argsort()]
    num_true = sum(conf_label_pair[:, 1])
    unique_conf = np.unique(conf_label_pair[:, 0])
    # print(conf_label_pair.shape)
    # print(unique_conf.shape)
    old_tp_rate = 0
    old_fp_rate = 0
    area_under_roc = 0
    for i, conf in enumerate(unique_conf):
        # print(conf)
        pred_true = conf_label_pair[conf_label_pair[:, 0] >= conf]
        tp_rate = sum(pred_true[:, 1] == True) / num_true
        fp_rate = sum(pred_true[:, 1] == False) / num_true
        if tp_rate > old_tp_rate and fp_rate > old_fp_rate:
            area_under_roc += (fp_rate - old_fp_rate) * tp_rate
            old_tp_rate = tp_rate
            old_fp_rate = fp_rate

    return avg_vals, std, area_under_roc
