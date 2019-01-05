import numpy as np
def hamming_loss(y_test, p_test):
    return 1.0 - (p_test==y_test).mean(axis=1)

def rank_loss(y_test, p_test):
    revloss = 1.0 * ((y_test==1) & (p_test==0)).sum(axis=1) * ((y_test==0) & (p_test==1)).sum(axis=1)
    eq0loss = 0.5 * ((y_test==1) & (p_test==0)).sum(axis=1) * ((y_test==0) & (p_test==0)).sum(axis=1)
    eq1loss = 0.5 * ((y_test==1) & (p_test==1)).sum(axis=1) * ((y_test==0) & (p_test==1)).sum(axis=1)
    return (revloss + eq0loss + eq1loss)

def f1_score(y_test, p_test):
    v1 = sum(2.0*(p_test*y_test))
    v2 = sum(p_test) + sum(y_test)
    v1[v2<=0] = 1.0
    v1[sum(y_test)<=0] = 1.0
    v1[v2>0] /= v2[v2>0]
    return v1

def accuracy_score(y_test, p_test):
    v1 = 1.0 * ((p_test==1) & (y_test==1)).sum(axis=1)
    v2 = 1.0 * ((p_test==1) | (y_test==1)).sum(axis=1) 
    v1[v2<=0] = 1.0
    v1[v2>0] /= v2[v2>0]
    return v1
class CFT:
    def __init__(self, cost, base_learner, params={}, n_round=8):
        self.cost = cost
        self.base_learner = base_learner
        self.params = params
        self.n_round = n_round
        if self.cost == 'ham':
            self.func = hamming_loss
        elif self.cost == 'rank':
            self.func = rank_loss
        elif self.cost == 'f1':
            self.func = f1_score
        elif self.cost == 'acc':
            self.func = accuracy_score

    def fit(self, x_train, y_train):
        try:
            self.K = y_train.shape[1]
        except:
            self.K = 1

        x_train_new = x_train
        p_train_new = y_train
        y_train_new = y_train
        w_train_new = self.cal_weight(y_train, y_train)
        
        for rd in xrange(self.n_round):
            self.clfs = [self.base_learner(**self.params) for i in xrange(self.K)]
            for i in xrange(self.K):
                self.clfs[i].fit(np.concatenate((x_train_new, p_train_new[:, : i]), axis=1), y_train_new[:, i], w_train_new[:, i])
            p_train = self.predict(x_train)

            w_train = self.cal_weight(y_train, p_train)
            x_train_new = np.concatenate((x_train_new, x_train), axis=0)
            p_train_new = np.concatenate((p_train_new, p_train), axis=0)
            y_train_new = np.concatenate((y_train_new, y_train), axis=0)
            w_train_new = np.concatenate((w_train_new, w_train), axis=0)
        #print w_train_new.mean()

    def predict(self, x_test):
        p_test = np.zeros((x_test.shape[0], self.K), dtype=int)
        for i in xrange(self.K):
            val = np.concatenate((x_test, p_test[:, :i]), axis=1)
            p_test[:, i] = np.array([self.clfs[i].predict(val[j, :]) for j in range(val.shape[0])])[:, 0]
        return p_test

    def cal_weight(self, y_train, p_train):
        
        score0 = np.zeros(y_train.shape)
        score1 = np.zeros(y_train.shape)
        
        for i in xrange(y_train.shape[1]):
            t_train = p_train.copy()
            t_train[:, i] = 0
            score0[:, i] = self.func(y_train, t_train)
            t_train[:, i] = 1
            score1[:, i] = self.func(y_train, t_train)
        w_train = np.abs(score0 - score1)
        w_train /= w_train.sum()
        w_train *= w_train.shape[0]*w_train.shape[1]
        return w_train
