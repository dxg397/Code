import os
import sys
from mldata import *
import math as m
import random as rand
import statistics
import numpy as np
import decimal as dec
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import mode

class _BinTree:
    # constructs a balanced binary tree
    # keeps track of which nodes compare which classes
    # node_comparisons -> [all nodes, nodes to the left]
    # childs -> [child left, child right]
        # terminal nodes are negative numbers
        # non-terminal nodes refer to the index in 'node_comparisons' for next comparison
    def __init__(self,n):
        self.n_arr=np.arange(n)
        print n,self.n_arr
        self.node_comparisons=[[None,None,None] for i in range(n-1)]
        self.node_counter=0
        self.childs=[[None,None] for i in range(n-1)]
        self.parents=[None for i in range(n-1)]
        self.isterminal=set()
        
        split_point=int(np.ceil(self.n_arr.shape[0]/2))
        self.node_comparisons[0][0]=list(self.n_arr)
        self.node_comparisons[0][1]=list(self.n_arr[:split_point])
        self.node_comparisons[0][2]=list(self.n_arr[split_point:])
        self.split_arr(self.n_arr[:split_point],0,True)
        self.split_arr(self.n_arr[split_point:],0,False)
        self.isterminal=list(self.isterminal)
        self.is_at_bottom=[i for i in range(len(self.childs)) if (self.childs[i][0]<=0) and (self.childs[i][1]<=0)]
        
    def split_arr(self,arr,parent_node,direction_left):
        if arr.shape[0]==1:
            if direction_left:
                self.childs[parent_node][0]=-arr[0]
            else:
                self.childs[parent_node][1]=-arr[0]
            self.isterminal.add(parent_node)
            return None
        
        self.node_counter+=1
        curr_node=self.node_counter
        if direction_left:
            self.childs[parent_node][0]=curr_node
        else:
            self.childs[parent_node][1]=curr_node
        self.parents[curr_node]=parent_node
        
        split_point=int(np.ceil(arr.shape[0]/2))
        self.node_comparisons[curr_node][0]=list(arr)
        self.node_comparisons[curr_node][1]=list(arr[:split_point])
        self.node_comparisons[curr_node][2]=list(arr[split_point:])
        self.split_arr(arr[:split_point],curr_node,True)
        self.split_arr(arr[split_point:],curr_node,False)
        return None

class FilterTree:
    """
    Parameters
    ----------
    base_classifier : object
        Base binary classification algorithm. Must have:
            * A fit method of the form 'base_classifier.fit(X, y, sample_weights = w)'.
            * A predict method.
    
    Attributes
    ----------
    nclasses : int
        Number of classes on the data in which it was fit.
    classifiers : list of objects
        Classifier that compares each two classes belonging to a node.
    tree : object
        Binary tree with attributes childs and parents.
        Non-negative numbers for children indicate non-terminal nodes,
        while negative and zero indicates a class (terminal node).
        Root is the node zero.
    base_classifier : object
        Unfitted base regressor that was originally passed.
    
    """
    def __init__(self, base_classifier):
        self.base_classifier=base_classifier

    
    def fit(self, X, C,y):
        """
        ----------
        X : array (n_samples, n_features)
            The data on which to fit a cost-sensitive classifier.
        C : array (n_samples, n_classes)
            The cost of predicting each label for each observation (more means worse).
        """
        X,C = self._check_fit_input(X,C)
        nclasses=C.shape[1]
        self.tree=_BinTree(nclasses)
        self.classifiers=[deepcopy(self.base_classifier) for c in range(nclasses-1)]
        classifier_queue=self.tree.is_at_bottom
        next_round=list()
        already_fitted=set()
        labels_take=-np.ones((X.shape[0],len(self.classifiers)))
        while True:
            for c in classifier_queue:
                if c in already_fitted or (c is None):
                    continue
                child1, child2 = self.tree.childs[c]
                if (child1>0) and (child1 not in already_fitted):
                    continue
                if (child2>0) and (child2 not in already_fitted):
                    continue
                    
                if child1<=0:
                    class1=-np.repeat(child1,X.shape[0]).astype('int64')
                else:
                    class1=labels_take[:, child1].astype('int64')
                if child2<=0:
                    class2=-np.repeat(child2,X.shape[0]).astype('int64')
                else:
                    class2=labels_take[:, child2].astype('int64')
                cost1=C[np.arange(X.shape[0]),np.clip(class1,a_min=0,a_max=None)]
                cost2=C[np.arange(X.shape[0]),np.clip(class2,a_min=0,a_max=None)]
 
                for nr in range (0,8):
                    y=(cost1<cost2).astype('uint8')
                    w=np.abs(cost1-cost2)

                    valid_obs=w>0
                    if child1>0:
                        valid_obs=valid_obs&(labels_take[:,child1]>=0)
                    if child2>0:
                        valid_obs=valid_obs&(labels_take[:,child2]>=0)
                    
                    X_take=X[valid_obs,:]
                    y_take=y[valid_obs]
                    w_take=w[valid_obs]
                    w_take=self._standardize_weights(w_take)
                    
                    self.classifiers[c].fit(X_take,y_take,sample_weight=w_take)
                    
                    labels_arr=np.c_[class1,class2].astype('int64')
                    pred = np.array([self.classifiers[c].predict(X_take[j, :]) for j in range(X_take.shape[0])])[:, 0]
                    labels_take[valid_obs,c]=labels_arr[np.repeat(0,X_take.shape[0]),pred.reshape(-1).astype('uint8')]
                    loss = 1 - (y==labels_take[valid_obs,c])
                    cost1=C[:,1]*loss
                    cost2=C[:,0]*loss
 
                    print  cost1,"############",nr
                already_fitted.add(c)
                next_round.append(self.tree.parents[c])
                if c==0 or (len(classifier_queue)==0):
                    break
            classifier_queue=list(set(next_round))
            next_round=list()
            if (len(classifier_queue)==0):
                break
        return self
    
    def _predict(self, X):
        curr_node=0
        while True:
            go_right=np.array([self.classifiers[curr_node].predict(X[j, :]) for j in range(X.shape[0])])[:, 0]
            if go_right:
                curr_node=self.tree.childs[curr_node][0]
            else:
                curr_node=self.tree.childs[curr_node][1]
                
            if curr_node<=0:
                return -curr_node

            
    def predict(self, X):
        """
        Predict the less costly class for a given observation
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict minimum cost label.
        method : str, either 'most-wins' or 'goodness':
            How to decide the best label (see Note)
        
        Returns
        -------
        y_hat : array (n_samples,)
            Label with expected minimum cost for each observation.
        """
        X=self._check_predict_input(X)
        if len(X.shape)==1:
            return self._predict(X.reshape(1, -1))
        elif X.shape[0]==1:
            return self._predict(X)
        else:
            out=list()
            for i in range(X.shape[0]):
                out.append(self._predict(X[i,:].reshape(1, -1)))
            return np.array(out)

    def _check_fit_input(self, X,C):
        if type(X)==pd.core.frame.DataFrame:
            X=X.as_matrix()
        if type(X)==np.matrixlib.defmatrix.matrix:
            X=np.array(X)
        if type(X)!=np.ndarray:
            raise ValueError("'X' must be a numpy array or pandas data frame.")
            
        if type(C)==pd.core.frame.DataFrame:
            C=C.as_matrix()
        if type(C)==np.matrixlib.defmatrix.matrix:
            C=np.array(C)
        if type(X)!=np.ndarray:
            raise ValueError("'C' must be a numpy array or pandas data frame.")
            
        assert X.shape[0]==C.shape[0]
        #assert C.shape[1]>2
        
        return X,C
    
    def _check_predict_input(self, X):
        if type(X)==pd.core.frame.DataFrame:
            X=X.as_matrix()
        if type(X)==np.matrixlib.defmatrix.matrix:
            X=np.array(X)
        if type(X)!=np.ndarray:
            raise ValueError("'X' must be a numpy array or pandas data frame.")
        return X

    def _standardize_weights(self, w):
        return w*w.shape[0]/w.sum()
