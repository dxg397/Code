#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""
import sys
import os
import numpy as np
import mldata
from NB_Model import NaiveBayes
from evaluation_lib import k_fold_cv

K_FOLD = 5

def main():
    """
    run the naive bayes network with given command line input
    ----------
    """
    file_path, use_full_sample, n_bins, m_estimate = sys.argv[1:5]
    # parse args
    [use_full_sample, n_bins, m_estimate] = [int(use_full_sample), int(n_bins), int(m_estimate)]
    examples = get_dataset(file_path)
    params = [n_bins, m_estimate]
    if use_full_sample:
        samples = examples[:, 1:-1]
        labels = examples[:, -1]
        n_b = NaiveBayes(*params)
        n_b.fit(samples, labels)
    else:
        avg_vals, std, area_under_roc = k_fold_cv('nbayes', params, examples, K_FOLD)
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
