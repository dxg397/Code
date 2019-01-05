#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import sys
import os
import numpy as np
import mldata
from DT_Model import ID3

K = 5 # number of folds

def main():
    """
    run the decision tree with param given by user
    ----------
    """
    file_path, use_full_sample, max_depth, use_gain_ratio = sys.argv[1:5]
    # parse args
    [use_full_sample, max_depth, use_gain_ratio] = [int(use_full_sample), int(max_depth), int(use_gain_ratio)]
    # parse dataset
    raw_parsed = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    examples = np.array(raw_parsed, dtype=object)
    samples = examples[:, 1:-1]
    targets = examples[:, -1]
    # grow a huge tree (gurantees to cover a full tree) if input specifies 0 in max_depth
    if max_depth == 0:
        max_depth = int(1e9)
    # run on full sample
    if use_full_sample:
        dt = ID3(max_depth, use_gain_ratio)
        dt.fit(samples, targets)
    else:
        dt = ID3(max_depth, use_gain_ratio)
        print("Accuracy: ", str(k_fold_cv(dt, examples, K)))
    print("Size: ", str(dt.size))
    print("Maximum Depth: ", str(dt.max_depth))
    print("First Feature: ", str(raw_parsed.examples[0].schema.features[dt.attr_idx+1].name))

def k_fold_cv(model, data, k):
    """
    perform k fold cross validation on the model
    ----------
    model : ID3
          an instance of ID3 to be cross validated
    data : array-like
          the entire dataset
    k : int
          the parameter in cross validation determing how many fold we're doing
    """
    data_split = np.array_split(data, k)
    acc = []
    for i in range(0, k):
        train_data = np.delete(data_split, (i), axis=0)
        train_data = np.concatenate(train_data)
        val_data = data_split[i]
        train_samples = train_data[:, 1:-1]
        train_targets = train_data[:, -1]
        val_samples = val_data[:, 1:-1]
        val_targets = [bool(x) for x in val_data[:, -1]]
        model.fit(train_samples, train_targets)
        pred = [bool(model.predict(val_samples[j, :])) for j in range(val_samples.shape[0])]
        acc.append(accuracy(val_targets, pred))
    return sum(acc) / float(k)

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

if __name__ == '__main__':
    main()
