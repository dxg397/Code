#!usr/bin/python3
"""
This is an implementation of logistic regression algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""
import sys
import os
import numpy as np
import mldata
from evaluation_lib import k_fold_cv
from LR_Model import LogisticRegression

K_FOLD = 5

def main():
    """
    run the logistic regression with given command line input
    ----------
    file_path = path of the dataset
    use_full_sample = if we want to train the data on the full dataset or we want to use it for cross-validation
    lr = learning rate of the model
    num_iter = number of iterations used for training
    _lambda = penalty variable
    """
    file_path, use_full_sample, lr, num_iter, _lambda = sys.argv[1:6]
    # parse args
    [use_full_sample, num_iter, lr,_lambda] = [int(use_full_sample), int(num_iter), float(lr), float(_lambda)]
    examples = get_dataset(file_path)
    params = [lr, num_iter, _lambda]
    if use_full_sample:
        samples = examples[:, 1:-1]
        labels = examples[:, -1]
        clf = LogisticRegression(*params)
        clf.fit(samples, labels)
    else:
        avg_vals, std, area_under_roc = k_fold_cv('logreg', params, examples, K_FOLD)
        print (("Accuracy: %.3f %.3f " + os.linesep +
                "Precision: %.3f %.3f " + os.linesep +
                "Recall: %.3f %.3f" + os.linesep +
                "Area under ROC: %.3f") % (avg_vals[0], std[0], avg_vals[1], std[1], avg_vals[2], std[2], area_under_roc))

def get_dataset(file_path):
    """
    parse the dataset stored in the input file path
    ----------
    file_path : String
        the path to the dataset
    """
    raw_parsed = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    return np.array(raw_parsed, dtype=object)

if __name__ == '__main__':
    main()
