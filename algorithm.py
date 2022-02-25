import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from loss import *
from metric import *

class Adaptive_clf_Abernethy():
    def __init__(self):
        self.clf = LogisticRegression()
        self.X_train = None
        self.y_train = None
        self.group_train = None
        
        self.X_val = None
        self.y_val = None
        self.group_val = None
        
        self.X_pool = None 
        self.y_pool = None
        self.group_pool = None
        
        self.X_test = None
        self.y_test = None
        self.group_test = None
        
        self.fairness_violation = []
        self.train_loss = []
        pass
    
    def make_set(self, feature, label, group ,train_size, val_size, feature_test, label_test, group_test):
        """
        Give feature set and corresponding label set to generate train set, sample pool, validation set, and test set.

        Args:
            feature (list/np.array): m x n matrix, 
            label (list/np.array): labels
            group (list/np.array): group label
            train_size: the proportion of training data
            val_size: the proportion of validation data
            (the rest is sample pool)
        """
        # print(feature)
        feature = np.array(feature)
        label = np.array(label)
        group = np.array(group)
        # print(feature.shape)
        # num_validation = int(len(label) * val_size)
        # num_train = int(len(label) * train_size)
        num_validation = val_size
        num_train = train_size
        
        group0_index = group == 0
        group1_index = group == 1
    
        group0_feature = feature[group0_index, :]
        group1_feature = feature[group1_index, :]

        group0_label = label[group0_index]
        group1_label = label[group1_index]
        
        
        group0_positive = group0_feature[group0_label == 1, :]
        group0_negative = group0_feature[group0_label == -1, :]
        group1_positive = group1_feature[group1_label == 1, :]
        group1_negative = group1_feature[group1_label == -1, :]
        
        group0_positive_train, group0_positive_val = train_test_split(group0_positive, test_size=(num_validation//4))
        group0_negative_train, group0_negative_val = train_test_split(group0_negative, test_size=(num_validation//4))
        group1_positive_train, group1_positive_val = train_test_split(group1_positive, test_size=(num_validation//4))
        group1_negative_train, group1_negative_val = train_test_split(group1_negative, test_size=(num_validation//4))
       
        self.X_val = np.vstack([group0_positive_val, group0_negative_val, group1_positive_val, group1_negative_val])
        self.y_val = np.hstack([np.ones(num_validation//4), (-1)* np.ones(num_validation//4), np.ones(num_validation//4), (-1)*np.ones(num_validation//4)])
        self.group_val = np.hstack([np.zeros((num_validation//4) * 2), np.ones((num_validation//4) * 2)])
        
        train_and_pool_feature = np.vstack([group0_positive_train, group0_negative_train, group1_positive_train, group1_negative_train])
        train_and_pool_group = np.hstack([np.zeros(len(group0_positive_train) + len(group0_negative_train)), np.ones((len(group1_positive_train) + len(group1_negative_train)))])
        train_and_pool_label = np.hstack([np.ones(len(group0_positive_train)), (-1)* np.ones(len(group0_negative_train)), np.ones(len(group1_positive_train)), (-1)*np.ones(len(group1_negative_train))])
        
        
        # self.X_train, self.X_val, self.y_train, self.y_val, self.group_train, self.group_val = train_test_split(feature, label, group,test_size=num_validation)
        self.X_pool, self.X_train, self.y_pool, self.y_train, self.group_pool, self.group_train = train_test_split(train_and_pool_feature, train_and_pool_label, train_and_pool_group,test_size=num_train)
        self.X_train = list(self.X_train)
        self.y_train = list(self.y_train)
        self.group_train = list(self.group_train)
        
        self.X_pool = list(self.X_pool)
        self.y_pool = list(self.y_pool)
        self.group_pool = list(self.group_pool)
        
        self.X_test = list(feature_test)
        self.y_test = list(label_test)
        self.group_test = list(group_test)
    def train(self, fairness, loss, p=0.5, T = 500):
        """
        train logistic regression according to the fairness metric.

        Args:
            train ([type]): [description]
            label ([type]): [description]
            fairness: fairness metrics for training
            T: sample budget
            p: prob of choosing next sample from the whole population
        """
        # train and record loss
        self.clf.fit(self.X_train, self.y_train)
        
        for i in range(1, T + 1):
            
            y_pred = self.clf.predict(self.X_test)
            train_error = loss(self.y_test, y_pred)
            # self.train_loss.append(train_error)
            # print(self.group_val)
            # print(self.y_val)
            # print(y_pred)
            self.train_loss.append(train_error)
            
            
            
            # record fairness violation
            if i % 500 == 0:
                # self.fairness_violation.append(np.abs(fairness_violation))
                demographic_parity = np.abs(Demographic_parity_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                equal_odds = np.abs(Equal_odds_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                equal_oppo = np.abs(Equal_opportunity_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                overall_acc = np.abs(Overall_Accuracy_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                # print(f"training error: {train_error}, demographic parity violation: {np.abs(demographic_parity)}, equal odds violation: {equal_odds}, equal opportunity violation: {equal_oppo}, overall accuracy violation: {overall_acc}")
                self.fairness_violation.append([p, i, train_error, demographic_parity, equal_odds, equal_oppo, overall_acc])
            
            y_pred = self.clf.predict(self.X_val)
            fairness_violation = fairness(np.array(self.group_val), np.array(y_pred), np.array(self.y_val))
            
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
        return self.fairness_violation, self.train_loss
            
    def test(self, fairness):
        """
        output test result--accuracy and fairness violation

        Args:
            fairness (method): fairness metric
        """
        y_pred = self.clf.predict(self.X_test)
        return metrics.accuracy_score(self.y_test, y_pred), fairness(self.group_test, y_pred, self.y_test)
        
    def display(self):
        plt.plot(self.train_loss)
        plt.show()
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
        
        self.X_test = None
        self.y_test = None
        self.group_test = None

        self.pi = [0, 0]
        self.N = [0, 0]
        self.ez = [0, 0]

        self.train_loss = []
        self.fairness_violation = []
    def make_set(self, feature, label, group, feature_test, label_test, group_test):
        """
        Give feature set and corresponding label set to generate train set, sample pool, validation set, and test set.

        Args:
            feature (list/np.array): m x n matrix, 
            label (list/np.array): labels
            group (list/np.array): group label
        """
        feature = np.array(feature)
        label = np.array(label)
        group = np.array(group)
        
        self.feature[0] = list(feature[group == 0])
        self.label[0] = list(label[group == 0])
        self.feature[1] = list(feature[group == 1])
        self.label[1] = list(label[group == 1])
        # add one sample from group0 to train set
        index = np.random.randint(0, len(self.label[0]))
        self.Dt.append(self.feature[0][index])
        self.Dt_label.append(self.label[0][index])
        self.Dt_group.append(0)
        label = self.label[0][index]
        self.feature[0].pop(index)
        self.label[0].pop(index)
        
        
        # add another sample from group0 (different label) to train set
        index = np.random.randint(0, len(self.label[0]))
        while (self.label[0][index] == label):
            index = np.random.randint(0, len(self.label[0]))
        self.Dt.append(self.feature[0][index])
        self.Dt_label.append(self.label[0][index])
        self.Dt_group.append(0)
        self.feature[0].pop(index)
        self.label[0].pop(index)

        # add one sample from group1 to train set
        index = np.random.randint(0, len(self.label[1]))
        self.Dt.append(self.feature[1][index])
        self.Dt_label.append(self.label[1][index])
        self.Dt_group.append(1)
        label = self.label[1][index]
        self.feature[1].pop(index)
        self.label[1].pop(index)

        
        # add another sample from group1 (different label) to train set
        index = np.random.randint(0, len(self.label[1]))
        while (self.label[1][index] == label):
            index = np.random.randint(0, len(self.label[1]))
        self.Dt.append(self.feature[1][index])
        self.Dt_label.append(self.label[1][index])
        self.Dt_group.append(1)
        self.feature[1].pop(index)
        self.label[1].pop(index)



        # add one sample from group0 to val set
        index = np.random.randint(0, len(self.label[0]))
        self.Dz[0].append(self.feature[0][index])
        self.Dz_label[0].append(self.label[0][index])
        label = self.label[0][index]
        self.feature[0].pop(index)
        self.label[0].pop(index)

        
        # add another sample from group0 (different label) to val set
        index = np.random.randint(0, len(self.label[0]))
        while (self.label[0][index] == label):
            index = np.random.randint(0, len(self.label[0]))
        self.Dz[0].append(self.feature[0][index])
        self.Dz_label[0].append(self.label[0][index])
        self.feature[0].pop(index)
        self.label[0].pop(index)

        # add one sample from group1 to val set
        index = np.random.randint(0, len(self.label[1]))
        self.Dz[1].append(self.feature[1][index])
        self.Dz_label[1].append(self.label[1][index])
        label = self.label[1][index]
        self.feature[1].pop(index)
        self.label[1].pop(index)

        
        # add another sample from group1 (different label) to val set
        index = np.random.randint(0, len(self.label[1]))
        while (self.label[1][index] == label):
            index = np.random.randint(0, len(self.label[1]))
        self.Dz[1].append(self.feature[1][index])
        self.Dz_label[1].append(self.label[1][index])
        self.feature[1].pop(index)
        self.label[1].pop(index)
        
        # update N and pi 
        self.N[0] = 1
        self.N[1] = 1
        self.pi[0] = 0.5
        self.pi[1] = 0.5


        self.X_test = list(feature_test)
        self.y_test = list(label_test)
        self.group_test = list(group_test)
        
    def train(self, n, loss, C=1):
        """
        train logistic regression according to the fairness metric.

        Args:
            n: budget
            loss: loss function
        """
        for t in range(1, (n//2) + 1):
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
            self.label[zt].pop(index)
            self.Dt_group.append(zt)

            index = np.random.randint(0, len(self.label[zt]))
            self.Dz[zt].append(self.feature[zt][index])
            self.feature[zt].pop(index)
            self.Dz_label[zt].append(self.label[zt][index])
            self.label[zt].pop(index)
            self.N[zt] += 1
            self.pi[0] = self.N[0] / (self.N[0] + self.N[1])
            self.pi[1] = self.N[1] / (self.N[0] + self.N[1])
            self.update_ez(zt)

            self.clf.fit(self.Dt, self.Dt_label)
                
            y_pred = self.clf.predict(self.X_test)
            train_error = loss(y_pred, self.y_test)
            self.train_loss.append(train_error)

            if t % 500 == 0:
                demographic_parity = np.abs(Demographic_parity_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                equal_odds = np.abs(Equal_odds_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                equal_oppo = np.abs(Equal_opportunity_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                overall_acc = np.abs(Overall_Accuracy_worst_group(np.array(self.group_test), np.array(y_pred), np.array(self.y_test)))
                # print(f"training error: {train_error}, demographic parity violation: {np.abs(demographic_parity)}, equal odds violation: {equal_odds}, equal opportunity violation: {equal_oppo}, overall accuracy violation: {overall_acc}")
                self.fairness_violation.append([C, t, train_error, demographic_parity, equal_odds, equal_oppo, overall_acc])
        return self.fairness_violation, self.train_loss


    def U_t(self, z, f, loss_func, C = 1):
        loss = 0.0
        y_pred = f.predict(self.Dz[z])
        loss = loss_func(self.Dz_label[z], y_pred)
            
        return 1/len(self.Dz_label[z])*loss + self.ez[z] + (2*C/self.pi[z]) * (self.pi[0]*self.ez[0] + self.pi[1]*self.ez[1])
    
    
    def update_ez(self, z, dvc=3, delta=0.05):
        e = 2*np.sqrt(2*dvc*np.log(2*np.e*self.N[z]/dvc) + 2*np.log(2*self.N[z]*np.pi*2*2/(3*delta)))
        self.ez[z] = e

        
    def plot(self):
        plt.plot(self.train_loss)
        plt.plot()
        pass