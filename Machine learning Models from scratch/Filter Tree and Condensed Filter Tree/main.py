import os
import sys
import mldata
import math as m
import random as rand
import statistics
import numpy as np
import decimal as dec
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from FT_MLC import FilterTree
from CFT_MLC import CFT
LR_Config = [0.01, 100, 0.1]
def main():

    file_path = sys.argv[1]
    clss = sys.argv[2]
    examples = get_dataset(file_path)
    samples = examples[:, 1:-1]
    targets = examples[:, -1]
    
    y=examples[:,examples.shape[1]-1]-0
    if clss in ["FT","FilterTree", "Filter Tree","filter tree"]:
        print ("Filter Tree Invoked.")
        ml=MultiLabelBinarizer()
        y_onehot=ml.fit_transform(y.reshape(-1,1))
        n_classes = y_onehot.shape[1]
        missclassif_cost_matrix=np.zeros((n_classes,n_classes))
        np.random.seed(1234)
        for i in range(n_classes-1):
            for j in range(i+1,n_classes):
                cost_missclassif=np.random.gamma(1,5)
                missclassif_cost_matrix[i,j]=0.25
                missclassif_cost_matrix[j,i]=0.9
        C = np.array([missclassif_cost_matrix[i] for i in y])
        costsensitive_FT = FilterTree(Weighted_LogisticRegression(*LR_Config))
        acc,t_cost = k_fold_cv_FT(costsensitive_FT,examples,C,5)
        print ("Accuracy:", acc)
        print ("Average accuracy: %f",acc.mean())
        print ("Total Cost:", t_cost)
        print ("Average total cost: %f", t_cost.mean())
    else:
        print ("Condensed Filter Tree Invoked.")
        params = {"learning_rate":0.01, "num_iter":100, "_lambda":0.1 }
        alg_cft_ham = CFT('ham', Weighted_LogisticRegression, params)
        alg_cft_acc = CFT('acc', Weighted_LogisticRegression, params)
        acc_b,t_cost = k_fold_cv(alg_cft_acc,alg_cft_ham,examples,5)
        print ("Accuracy:", acc)
        print ("Average accuracy: %f",acc.mean())
        print ("Total Cost:", t_cost)
        print ("Average total cost: %f", t_cost.mean())





def get_dataset(file_path):
    """
    parse the dataset stored in the input file path
    ----------
    file_path : String
        the path to the dataset
    """
    raw_parsed = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    return np.array(raw_parsed, dtype=object)

def k_fold_cv_FT(model, data,cost, k):
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
    #print data.shape,cost.shape
    data_split = np.array_split(data, k)
    cost_split = np.array_split(cost, k)
    acc = []
    test_cost = []

    np.random.seed(12345)
    np.random.shuffle(data)
    for i in range(0, k):
        train_data = np.delete(data_split, (i), axis=0)
        train_data = np.concatenate(train_data)
        val_data = data_split[i]
        
        C_train = np.delete(cost_split, (i), axis=0)
        C_train = np.concatenate(C_train)
        C_test = cost_split[i]
       # print C_train.shape
        train_samples = train_data[:, 1:-1]
        #print train_samples.shape
        train_targets = train_data[:,train_data.shape[1]-1]-0
        val_samples = val_data[:, 1:-1]
        val_targets = [bool(x) for x in val_data[:, -1]]
        
        model.fit(train_samples, C_train,train_targets)
        pred = model.predict(val_samples)

        acc.append(accuracy(val_targets, pred))
        test_cost.append(C_test[np.arange(C_test.shape[0]), pred].sum())

    return acc,test_cost

def k_fold_cv_CFT(model,model1, data, k):
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
    ham = []
    rank = []
    f1=[]
    np.random.seed(12345)
    np.random.shuffle(data)
    for i in range(0, k):
        train_data = np.delete(data_split, (i), axis=0)
        train_data = np.concatenate(train_data)
        val_data = data_split[i]
        train_samples = train_data[:, 1:-1]
        train_targets = train_data[:, -1]
        train_targets = train_targets.reshape(-1,1)
        val_samples = val_data[:, 1:-1]
        val_targets = [bool(x) for x in val_data[:, -1]]
        model.fit(train_samples, train_targets)
        pred = model.predict(val_samples)
        acc.append(accuracy(val_targets, pred))
        
        model1.fit(train_samples, train_targets)
        pred = model1.predict(val_samples)
        ham.append(hamming_loss(val_targets, pred).mean())

    return acc,ham


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
