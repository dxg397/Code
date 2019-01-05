#!usr/bin/python3
"""
This is an implementation of naive Bayes algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

class NaiveBayes(object):
    """
    a naive bayes model
    """

    def __init__(self, n_bins, m_estimate):
        self.n_bins = n_bins
        self.m_estimate = m_estimate
        # store bins
        self.bins = {}
        # Pr(Y = True)
        self.p_true = -1
        # prior estimate used in calculating likelihood
        self.prior_e = -1
        # value of m used in calculating likelihood
        self.m_val = -1
        # probability dictionary
        self.prob_dict = {}
        self.prob_dict[True] = {}
        self.prob_dict[False] = {}

    def fit(self, samples, labels):
        """
        build a naive bayes network with input samples and labels
        ----------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        for attr_idx, attr in enumerate(samples.T):
            if isinstance(attr[0], float):
                self.bins[attr_idx] = np.linspace(min(attr)-1, max(attr)+1, num=self.n_bins + 1)
                attr = np.digitize(attr.astype(np.float64), self.bins[attr_idx])
            else:
                self.bins[attr_idx] = list(np.unique(attr))
            al_pair = np.array([attr, labels]).T
            probs = self.likelihood(al_pair, len(self.bins[attr_idx]))
            self.prob_dict[True][attr_idx] = probs[:, 0]
            self.prob_dict[False][attr_idx] = probs[:, 1]
        self.p_true = sum(labels) / float(len(labels))

    def predict(self, _x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        _x : array-like
            the sample data
        """
        if self.p_true == -1:
            raise Exception("Fit the data before making prediction")
        p_pos, p_neg = [self.p_true, (1 - self.p_true)]
        for attr_idx, val in enumerate(_x):
            if isinstance(val, float):
                bin_idx = self.locate_idx(val, self.bins[attr_idx]) - 1
            else:
                bin_idx = list(self.bins[attr_idx]).index(val)
            p_pos *= self.prob_dict[True][attr_idx][bin_idx]
            p_neg *= self.prob_dict[False][attr_idx][bin_idx]

        confidence = p_pos / (p_pos + p_neg)
        return bool(p_pos >= p_neg), confidence

    def locate_idx(self, val, sorted_bin):
        """
        find input value's corresponding bin index in the sorted bin list
        ----------
        val : int
            the value to be looked up
        sorted_bin : array-like
            the bin list
        """
        # return index of last bin if val >= upper bound
        if val >= max(sorted_bin):
            return len(sorted_bin) - 1
        # return index of first bin if val < lower bound
        elif val < min(sorted_bin):
            return 1
        # return corresponding bin index if val is between in [lower bound, upper bound)
        else:
            for i, curr_val in enumerate(sorted_bin[:-1]):
                if val >= curr_val and val < sorted_bin[i+1]:
                    return i + 1

    def likelihood(self, al_pair, _v):
        """
        calculates the likelihood of attribute taking on a unique value given its label
        ----------
        al_pair : array-like
            the attribute and label pair
        _v : int
            number of unique value of input attribute
        return : array-like
            array of conditional probabilities with shape [_v, 2]
        """
        # check if we're doing Laplace smoothing
        if self.m_estimate < 0:
            self.m_val = _v
            self.prior_e = 1/float(_v)
        else:
            self.m_val = self.m_estimate
            self.prior_e = self.p_true
        # unique value and its corresponding conditional probability
        lh_array = np.zeros((_v, 2))
        for idx, uniq_val in enumerate(np.unique(al_pair[:, 0])):
            lh_array[idx, 0] = self.compute_lh(al_pair[al_pair[:, 1] == True], uniq_val, self.prior_e)
            lh_array[idx, 1] = self.compute_lh(al_pair[al_pair[:, 1] == False], uniq_val, 1 - self.prior_e)
        return lh_array

    def compute_lh(self, al_pure_pair, x_i, p):
        """
        computes the likelihood of attribute taking on xi as value given its label
        ----------
        al_pair : array-like
            the attribute and label pair
        """
        # (number of examples with Xi = xi and Y = y) + mp
        numerator = len(al_pure_pair[al_pure_pair[:, 0] == x_i]) + self.m_val * p
        # (number of examples with Y = y) + m
        denominator = len(al_pure_pair) + self.m_val
        return numerator / float(denominator)
