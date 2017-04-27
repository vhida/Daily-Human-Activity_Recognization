# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:22:11 2017

@author: Jian Yang local
"""

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from random import shuffle

class OneVSallSVM():
    """
    One vs All Ensemble classifier with SVC as core
    """
    def __init__(self):
        self.data = np.genfromtxt("../data/csv_data.csv", delimiter = ',')
        self.clf =  OneVsRestClassifier(SVC(kernel='linear'))
        self.DATA_N = self.data.shape[0]
        self.TRAIN_N = int(np.floor(self.DATA_N/3))
        
        
    def run(self):
        """
        Initialization with following parameters:
        data: data matrix
        DATA_N: total number of data rows
        TRAIN_N: data is sliced into 3 parts, 2/3 will be used as training
            and the rest for validation
        score: validation accuracy
        """
        index = np.arange(self.DATA_N)
        shuffle(index)
        
        print index.shape, self.TRAIN_N
        
        train_index = index[0:self.TRAIN_N]
        valid_index = index[self.TRAIN_N:]

        print train_index.shape, valid_index.shape

        X = self.data[:,:-1]
        Y = self.data[:, -1]

        train_x = X[train_index]
        train_y = Y[train_index]
        
        valid_x = X[valid_index]
        valid_y = Y[valid_index]

        self.clf.fit(train_x, train_y)
#        self.predict(valid_x, valid_y)
        
        print self.clf.score(valid_x, valid_y)
        
ova = OneVSallSVM()
ova.run()      
        
