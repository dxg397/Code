#!usr/bin/python3
"""
This is an implementation of bagging algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np

class Bagging(object):
    """
    a bagging classifier
    """

    def __init__(self, base_classifier, n_iter):
        self.n_iter = n_iter
        self.classifiers = [base_classifier]*n_iter


    def ensemble_fit(self, samples, labels):
        """
        Build a Bagging ensemble of classifier from the training
           set (X, y) using bootstrap
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values
        """
        
        for i in self.n_iters:
            new_samples , new_targets = self.random_sample(samples,labels)
            self.classifiers[i].fit(new_samples, new_targets)

    def ensemble_predict(self, sample):
        """
        Predict class for X.
        The predicted class of an input sample is computed by majority vote
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        predictions = []
        for j in range(sample.shape[0]):
            pred = []
            for cla in self.classifiers:
                pred.append(bool(cla.predict(sample[j, :])))
            predictions.append(sum(pred)/len(pred) >= 0.5)
        return predictions
        

    def random_samples(self,samples,labels, ratio=1.0):
        """
        Generates and returns a randomised sample dataset same
        as the size of the initial one and
        with replacemnet
        """
        rand_sample = list()
        rand_label = list()
        n_sample = round(len(samples) * ratio)
        while len(rand_sample) < n_sample:
            index = randrange(len(rand_sample))
            rand_sample.append(samples[index])
            rand_label.append(labels[index])
        return np.array(rand_sample), np.array(rand_label)
