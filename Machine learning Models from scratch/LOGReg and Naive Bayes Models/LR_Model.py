#!usr/bin/python3
"""
This is an implementation of logistic regression algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

import numpy as np
from scipy.stats import truncnorm

class LogisticRegression(object):
    """
    a logistic regression model
    """

    def __init__(self, learning_rate, num_iter, _lambda):
        self._lambda = _lambda
        self.weights = None
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.loss = None
        self.attr_dict={}

    def fit(self, samples, labels):
        """
        build a Logistic Regression  with input samples and labels
        ----------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        for i in range(samples.shape[1]):
            if not isinstance(samples[:,i][0], float):
                samples[:, i], self.attr_dict[i] = self.encode(samples[:, i])
        # self.weights = np.zeros(len(samples[0]))
        self.weights = truncnorm.rvs(-1, 1, loc=0, scale=1, size=samples.shape[1])
        self.bias = 0
        for i in range(self.num_iter):
            gradient_weights, gradient_bias = self.gradient(samples, labels)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        X : array-like
            the sample data
        """
        encoded_x = np.zeros(len(x))
        for i, val in enumerate(x):
            if not isinstance(val, float):
                encoded_x[i] = self.attr_dict[i][val]
            else:
                encoded_x[i] = val
        log_probab = np.dot(self.weights , encoded_x) + self.bias
        confidence = self.sigmoid(encoded_x)
        return bool(log_probab > 0), confidence

    def sigmoid(self, samples):
        '''
        logistic(sigmoid) function
        '''
        x = -(np.dot(self.weights, samples) + self.bias)
        return 1.0 / (1 + np.exp(x))

    def encode(self, attrs):
        """
        encode nominal attributes to 1-k
        :param attrs: array-like
            attribute vector
        :return:
        """
        transdict = {}
        unique_vals = np.unique(attrs)
        for i, val in enumerate(unique_vals):
            transdict[val] = float(i)
        return list(map(lambda x: transdict[x], attrs)),transdict

    def gradient(self, samples, labels):
        """
        :param samples: array-like
            Xs
        :param labels: array-like
            Ys
        :return: derivative_of_weights : array-like
            vector of weights' derivative
                derivative_of_bias : float
            derivative of bias
        """
        derivative_of_weights = np.zeros(samples.shape[1])
        derivative_of_bias = 0
        for i, X in enumerate(samples.astype(np.float64)):
            cond_likelihood = self.sigmoid(X)
            if labels[i] == True:
                derivative_of_weights += (cond_likelihood - 1) * X
                derivative_of_bias += cond_likelihood - 1
            else:
                derivative_of_weights += cond_likelihood * X
                derivative_of_bias += cond_likelihood

        derivative_of_weights += self._lambda * self.weights
        return derivative_of_weights, derivative_of_bias
