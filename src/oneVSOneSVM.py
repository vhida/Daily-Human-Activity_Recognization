# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:09:51 2017

@author: Jian Yang local
"""
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from random import shuffle



class OneVSOneSVM():
    """
    One vs One Ensemble classifier with SVC as core
    """
    
    def __init__(self):
        """
        Initialization with following parameters:
            data: data matrix
            DATA_N: total number of data rows
            TRAIN_N: data is sliced into 3 parts, 2/3 will be used as training
                and the rest for validation
            score: validation accuracy
        """
        self.data = np.genfromtxt("../data/processed_data.csv", delimiter = ' ')
        self.clf =  OneVsOneClassifier(SVC(kernel='linear'))
        self.DATA_N = self.data.shape[0]
        self.TRAIN_N = int(np.floor(self.DATA_N/3))
        self.SCORE = 0
        
    def __str__(self):
        return "the score for ONeVSOneSVM is: " + str(self.score)
        
    def run(self):
        """
        slice data and fit model, calculate score
        """
        index = np.arange(self.DATA_N)
        shuffle(index)
        
        
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
        self.score= self.clf.score(valid_x, valid_y)
        
        
        
ovo = OneVSOneSVM()
ovo.run() 
print ovo