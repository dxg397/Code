#!usr/bin/python3
"""
This is an implementation of ID3 decision tree https://en.wikipedia.org/wiki/ID3_algorithm
Author: Stan Tian, Yimin Chen, Devansh Gupta
"""

from collections import Counter
import numpy as np

REMOVE_ATTRIBUTE = True
IG_THRESHOLD = 0.0

class ID3(object):
    """
    a ID3 decision tree
    """

    def __init__(self, curr_depth, use_gain_ratio):
        self.curr_depth = curr_depth
        self.use_gain_ratio = use_gain_ratio
        # create 2 ID3 branches
        self.pos_branch = None
        self.neg_branch = None
        self.attr_idx = None
        self.part_val = None
        self.max_depth = 1
        self.size = 1
        self.feature = None

    def fit(self, samples, labels):
        """
        build a id3 decision tree with input samples and labels
        ---------
        samples : array-like
            the samples
        labels : array-like
            the labels
        """
        # no labels:
        if len(labels) == 0:
            self.attr_idx = -1
            self.max_depth = 0
            return

        # base case: max depth reached / pure node / run out of attributes
        if self.curr_depth == 0 or self.entropy_of(labels) == 0 or np.size(samples, 1) == 0:
            # create a leaf node with major label, id=-1 indicates leaf node
            self.attr_idx = -1
            self.max_depth = 0
            self.part_val = self.major_label(labels)
            return

        # recursive case: build subtrees
        self.attr_idx, self.part_val = self.best_attr_of(samples, labels)

        # early stopping: IG == 0
        if self.attr_idx == -1:
            self.part_val = self.major_label(labels)
            self.max_depth = 0
            return

        # partition the samples and labels
        pos_subs, neg_subs, pos_labels, neg_labels = self.partition(samples, labels, self.attr_idx, self.part_val)

        # init two branches
        self.pos_branch = ID3(self.curr_depth - 1, self.use_gain_ratio)
        self.neg_branch = ID3(self.curr_depth - 1, self.use_gain_ratio)

        # recursively build tree
        self.pos_branch.fit(pos_subs, pos_labels)
        self.neg_branch.fit(neg_subs, neg_labels)

        self.size += self.pos_branch.size + self.neg_branch.size
        self.max_depth += max(self.pos_branch.max_depth, self.neg_branch.max_depth)

    def predict(self, x):
        """
        predict the input instance's class label
        can only predict one sample at a time
        ----------
        samples : array-like
            the sample data
        """
        if self.attr_idx == -1:
            return self.part_val

        attr = x[self.attr_idx]
        if isinstance(attr, float):
            if attr <= self.part_val:
                return self.pos_branch.predict(np.delete(x, self.attr_idx))
            return self.neg_branch.predict(np.delete(x, self.attr_idx))
        if attr == self.part_val:
            return self.pos_branch.predict(np.delete(x, self.attr_idx))
        return self.neg_branch.predict(np.delete(x, self.attr_idx))

    def best_attr_of(self, samples, labels):
        """
        select the best attribute (give max information gain if chosen) from the input samples
        ----------
        samples : array-like
            the sample data
        """
        best_ig = 0.0
        best_attr_idx = None
        for i, attr in enumerate(samples.T):
            if self.use_gain_ratio:
                curr_ig, curr_partition = self.gr_of(attr, labels)
            else:
                curr_ig, curr_partition = self.ig_of(attr, labels)

            if best_ig <= curr_ig:
                best_ig = curr_ig
                best_partition = curr_partition
                best_attr_idx = i

        if best_ig <= IG_THRESHOLD:
            best_attr_idx = -1
        return best_attr_idx, best_partition

    def ig_of(self, attr, labels):
        """
        calculates the information gain if data partitioned by input attr
        ----------
        attr : array-like
            a list of values of a single attribute
        labels : array-like
            a list of values of labels
        """
        if isinstance(attr[0], float):
            # attr is continuous
            return self.ig_of_cont_attr(attr, labels)
        # attr is discrete or boolean
        return self.ig_of_discrete_attr(attr, labels)

    def ig_of_discrete_attr(self, attr, labels):
        """
        calculates the information gain of input attribute
        ----------
        attr : array-like
            the attribute column
        labels : array-like
            the class label column
        """
        unique_symbol = np.unique(attr)
        best_ig = 0.0
        for symbol in unique_symbol:
            curr_ig = self.ig_discrete(attr, symbol, labels)
            if best_ig <= curr_ig:
                best_ig = curr_ig
                best_symbol = symbol
        return best_ig, best_symbol

    def ig_discrete(self, attr, symbol, labels):
        """
        calculate the IG of the input symbol in a discrete attribute
        ----------
        attr : array-like
            the attribute column
        symbol : object
            a value in the discrete attribute
        labels : array-like
            the class label column
        """
        og_ent = self.entropy_of(labels)
        xy_pair = np.array([attr, labels]).T

        sym = xy_pair[xy_pair[:, 0] == symbol]
        non_sym = xy_pair[xy_pair[:, 0] != symbol]
        # calculate probability of current symbol
        p_sym = len(sym) / float(len(xy_pair))
        # calculate entropy of entire attribute using current symbol
        curr_ent = self.entropy_of(sym[:, 1]) * p_sym + self.entropy_of(non_sym[:, 1]) * (1 - p_sym)
        curr_ig = og_ent - curr_ent

        return curr_ig

    def ig_of_cont_attr(self, attr, labels):
        """
        calculates entropy of input continuous labels
        ----------
        attr : array-like
            the attribute column
        labels : array-like
            the class label column
        """
        sorted_attr, sorted_label, changed_idx = self.find_change_samples(attr, labels)

        og_ent = self.entropy_of(sorted_label)
        best_ig = 0.0
        for i in changed_idx[1:]:
            curr_ig = og_ent - self.entropy_cont_part(sorted_label, i)
            part_val = (sorted_attr[i] + sorted_attr[i-1])/2
            # Finding the maximum information gain
            if best_ig <= curr_ig:
                best_ig = curr_ig
                best_partition = part_val
        return best_ig, best_partition

    def entropy_cont_part(self, sorted_labels, part_idx):
        """
        :param sorted_labels:  array-like
            sorted according to the continuous attribute under consideration
        :param part_idx:  integer index
            partion the labels between index i-1 and index i
        :return:
        """
        length = len(sorted_labels)
        prob_left = part_idx / float(length)
        curr_ent = prob_left * self.entropy_of(sorted_labels[0:part_idx]) + (1 - prob_left) * self.entropy_of(sorted_labels[part_idx:length])
        return curr_ent

    def find_change_samples(self, attr, labels):
        """
        sort the attribute-label pairs according to attribute values in ascending order;
        find the indexs where the labels changes.
        :param attr: array-like, continuous
        :param labels: array-like
        :return: sorted attributes, sorted, labels, change indexs
        """
        cont_xy_pair = np.array([attr, labels]).T
        # Sort the attribute label ascendingly
        sorted_xy_pair = cont_xy_pair[cont_xy_pair[:, 0].argsort(kind='mergesort')]
        sorted_attr = sorted_xy_pair[:, 0]
        sorted_label = sorted_xy_pair[:, 1]
        # list of the indexes of samples where class label changed
        changed_idx = np.where(sorted_label != np.roll(sorted_label, 1))[0]

        return sorted_attr, sorted_label, changed_idx

    def gr_of(self, attr, labels):
        """
        calculates the gain ratio of input attribute
        ----------
        attr : array-like
            a sorted list of values of a single attribute
        labels : array-like
            a list of values of labels
        """
        if isinstance(attr[0], float):
            # attr is continuous
            return self.gr_of_cont_attr(attr, labels)
        # attr is discrete or boolean
        return self.gr_of_discrete_attr(attr, labels)

    def gr_of_discrete_attr(self, attr, labels):
        """
        calculates the information gain of input attribute
        ----------
        attr : array-like
            the attribute column
        labels : array-like
            the class label column
        """
        unique_symbol = np.unique(attr)
        best_gr = 0.0
        for symbol in unique_symbol:
            curr_ig = self.ig_discrete(attr, symbol, labels)
            curr_entrp = self.entropy_of_discrete(attr, symbol)
            if curr_entrp == 0:
                curr_gr = 0.0
            else:
                curr_gr = curr_ig / float(curr_entrp)
            if best_gr <= curr_gr:
                best_gr = curr_gr
                best_symbol = symbol
        return best_gr, best_symbol

    def gr_of_cont_attr(self, attr, labels):
        """
        calculates entropy of input continuous labels
        ----------
        attr : array-like
            the attribute column
        labels : array-like
            the class label column
        """
        sorted_attr, sorted_label, changed_idx = self.find_change_samples(attr, labels)

        og_ent = self.entropy_of(sorted_label)
        best_gr = 0.0
        for i in changed_idx[1:]:
            part_val = (sorted_attr[i] + sorted_attr[i - 1]) / 2
            curr_ig = og_ent - self.entropy_cont_part(sorted_label, i)
            curr_entrp = self.entropy_of_cont(sorted_attr, part_val)
            if curr_entrp == 0:
                curr_gr = 0.0
            else:
                curr_gr = curr_ig / float(curr_entrp)
            # Finding the maximum information gain
            if best_gr <= curr_gr:
                best_gr = curr_gr
                best_partition = part_val
        return best_gr, best_partition

    def entropy_of(self, labels):
        """
        calculates entropy of input labels
        ----------
        labels : array-like
            a list of labels
        """
        occurence = list(Counter(labels).values())
        prob = [x/float(np.sum(occurence)) for x in occurence]
        return -np.sum([x*np.log2(x) for x in prob])

    def entropy_of_discrete(self, attr, symbol):
        """
        calculates the entropy of choosing the input symbol in a discrete attribute
        ----------
        attr : array-like
            the attribute column
        symbol : object
            a value in the discrete attribute
        """
        sym = attr[attr[:] == symbol]
        p_sym = len(sym)/float(len(attr))
        if p_sym == 1 or p_sym == 0:
            return 0
        entropy = -p_sym*np.log2(p_sym) - (1-p_sym)*np.log2(1-p_sym)
        return entropy

    def entropy_of_cont(self, attr, part_val):
        """
        calculates the entropy of choosing the input symbol in a continuous attribute
        ----------
        attr : array-like
            the attribute column
        part_val : float
            the value we partition the data by
        """
        positive = attr[attr[:] <= part_val]
        p_positive = len(positive)/float(len(attr))
        if p_positive == 1 or p_positive == 0:
            return 0
        entropy = -p_positive * np.log2(p_positive) - (1 - p_positive) * np.log2(1 - p_positive)
        return entropy

    def partition(self, samples, labels, attr_idx, part_value):
        """
        partitions the samples and labels by input attribute and partition value
        ----------
        samples : array-like
            the sample data
        labels : array-like
            the label data
        attr_idx: int
            the index of the selected attribute taken as node
        part_value : float or String
            the mid value we partition data with
        """

        # get the indexs of samples which are positive according to the partition
        if isinstance(samples[0, attr_idx], float):
            index = np.where(samples[:, attr_idx] <= part_value)[0]
        else:
            index = np.where(samples[:, attr_idx] == part_value)[0]

        # get the to subset of samples by positiveness and negativeness
        pos_subs = samples[index]
        neg_subs = np.delete(samples, index, axis=0)
        pos_labels = labels[index]
        neg_labels = np.delete(labels, index, axis=0)
        # remove attribute
        if REMOVE_ATTRIBUTE:
            pos_subs = np.delete(pos_subs, attr_idx, axis=1)
            neg_subs = np.delete(neg_subs, attr_idx, axis=1)

        return pos_subs, neg_subs, pos_labels, neg_labels

    def major_label(self, labels):
        """
        return the label that is majority in the leaf node
        ----------
        labels : array-like
            the label column
        """
        keys = list(Counter(labels).keys())
        # if only contain one class
        if len(keys) == 1:
            return keys[0]
        # if contains nothing (anomaly
        elif not keys:
            raise Exception("EMPTY_LABELS_ERROR")
        counts = list(Counter(labels).values())
        if counts[0] > counts[1]:
            return keys[0]
        return keys[1]
