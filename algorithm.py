import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from loss import *

class Adaptive_clf_Abernethy():
    def __init__(self):
        self.clf = LogisticRegression()
        self.X_train = None
        self.y_train = None
        self.group_train = None
        
        self.X_test = None
        self.y_test = None
        self.group_test = None
        
        self.X_val = None
        self.y_val = None
        self.group_val = None
        
        self.X_pool = None 
        self.y_pool = None
        self.group_pool = None
        
        self.loss = []
        self.fairness_violation = []
        pass
    
    def make_set(self, feature, label, group ,train_size, val_size, test_size):
        """
        Give feature set and corresponding label set to generate train set, sample pool, validation set, and test set.

        Args:
            feature (list/np.array): m x n matrix, 
            label (list/np.array): labels
            group (list/np.array): group label
            train_size: the proportion of training data
            val_size: the proportion of validation data
            test_size: the proportion of test data
            (the rest is sample pool)
        """
        num_test = int(len(label) * test_size)
        num_validation = int(len(label) * val_size)
        num_train = int(len(label) * train_size)
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.group_train,self.group_test = train_test_split(feature, label, group,test_size=num_test)
        self.X_train, self.X_val, self.y_train, self.y_val, self.group_train, self.group_val = train_test_split(self.X_train, self.y_train, self.group_train,test_size=num_validation)
        self.X_pool, self.X_train, self.y_pool, self.y_train, self.group_pool, self.group_train = train_test_split(self.X_train, self.y_train, self.group_train,test_size=num_train)

    def train(self, fairness, loss, p=0.5, T = 5000):
        """
        train logistic regression according to the fairness metric.

        Args:
            train ([type]): [description]
            label ([type]): [description]
            fairness: fairness metrics
        """
        # train and record loss
        self.clf.fit(self.X_train, self.y_train).score(self.X_val, self.y_val)
        
        for i in range(T):
            
            y_pred = self.clf.predict(self.X_val)
            self.loss.append(loss(self.y_val, y_pred))
            fairness_violation = fairness(self.group_val, y_pred, self.y_val)
           
            # record fairness violation
            if i % 500 == 0:
                self.fairness_violation.append(np.abs(fairness_violation))
            
            # determine the disadvantaged group
            if (fairness_violation < 0):
                disadv_group = 0
            else: 
                disadv_group = 1
            # choose where to sample 
            if np.random.random_sample() < p:
                index = np.random.randint(0, len(self.y_pool))
                self.X_train.append(self.X_pool[index])
                self.y_train.append(self.y_pool[index])
                self.group_train.append(self.group_pool[index])
                
                self.X_pool.pop(index)
                self.y_pool.pop(index)
                self.group_pool.pop(index)
            else:
                index = np.random.randint(0, len(self.y_pool))
                while self.group_pool[index] != disadv_group:
                    index = np.random.randint(0, len(self.y_pool))
                self.X_train.append(self.X_pool[index])
                self.y_train.append(self.y_pool[index])
                self.group_train.append(self.group_pool[index])
                
                self.X_pool.pop(index)
                self.y_pool.pop(index)
                self.group_pool.pop(index)
            # train again using new training set
            self.clf.fit(self.X_train, self.y_train)
            
    def test(self, fairness):
        """
        output test result--accuracy and fairness violation

        Args:
            fairness (method): fairness metric
        """
        y_pred = self.clf.predict(self.X_test)
        return metrics.accuracy_score(self.y_test, y_pred), fairness(self.group_test, y_pred, self.y_test)
        
    def plot(self):
        pass
    
    
    
class Adaptive_clf_Shekhar():
    def __init__(self):
        self.clf = LogisticRegression()

        self.feature = [[],[]]
        self.label = [[], []]

        self.Dt = []
        self.Dt_label = []
        self.Dt_group = []

        self.Dz = [[], []]
        self.Dz_label = [[], []]

        self.pi = [0, 0]
        self.N = [0, 0]
        self.ez = [0, 0]

        self.loss = []
        self.fairness_violation = []
    
    def make_set(self, feature, label, group):
        """
        Give feature set and corresponding label set to generate train set, sample pool, validation set, and test set.

        Args:
            feature (list/np.array): m x n matrix, 
            label (list/np.array): labels
            group (list/np.array): group label
        """
        self.feature[0] = feature[group == 0]
        self.label[0] = label[group == 0]
        self.feature[1] = feature[group == 1]
        self.label[1] = label[group == 1]

    def train(self, n, loss, C=1):
        """
        train logistic regression according to the fairness metric.

        Args:
            n: budget
            loss: loss function
        """
        for t in range(n/2):
            if t < 2:
                zt = t
            elif min(self.N) < np.sqrt(t):
                zt = np.argmin(self.N)
            else:
                u0 = self.U_t(0, self.clf, loss, C=C)
                u1 = self.U_t(1, self.clf, loss, C=C)
                if u0 > u1:
                    zt = 0
                else:
                    zt = 1

            index = np.random.randint(0, len(self.label[zt]))
            self.Dt.append(self.feature[zt][index])
            self.feature[zt].pop(index)
            self.Dt_label.append(self.label[zt][index])
            self.label[t].pop(index)
            self.Dt_group.append(zt)

            index = np.random.randint(0, len(self.label[zt]))
            self.Dz[zt].append(self.feature[zt][index])
            self.group_feature.pop(index)
            self.Dz_label[zt].append(self.abel[zt][index])
            self.label[zt].pop(index)
            self.N[zt] += 1
            self.pi[0] = self.N[0] / (self.N[0] + self.N[1])
            self.pi[1] = self.N[1] / (self.N[0] + self.N[1])
            self.update_ez(zt)
            self.clf.fit(self.Dt, self.Dt_label)


    def U_t(self, z, f, loss_func, C = 1):
        loss = 0
        for i in range(len(self.Dz_label[0])):
            y_pred = f.predict(self.Dz[i])
            loss += loss_func(self.Dz_label[i], y_pred)
            
        return 1/len(self.Dz_label[0])*loss + self.ez[z] + (2*C/self.pi[z]) * (self.pi[0]*self.ez[0] + self.pi[1]*self.ez[1])
    
    
    def update_ez(self, z, dvc=3, delta=0.05):
        e = 2*np.sqrt(2*dvc*np.log(2*np.e*self.N[z]/dvc) + 2*np.log(2*self.N[z]*np.pi*2*2/(3*delta)))
        self.ez[z] = e

            
    # def test(self, fairness):
    #     """
    #     output test result--accuracy and fairness violation

    #     Args:
    #         fairness (method): fairness metric
    #     """
    #     y_pred = self.clf.predict(self.X_test)
    #     return metrics.accuracy_score(self.y_test, y_pred), fairness(self.group_test, y_pred, self.y_test)
        
    def plot(self):
        pass