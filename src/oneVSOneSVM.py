# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:09:51 2017

@author: Jian Yang local
"""
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



class OneVSOneSVM():
    """
    One vs One Ensemble classifier with SVC as core
    """
    
    def __init__(self):
        """
        Initialization with following parameters:
            data: data matrix
            
            score: validation accuracy
            pred_y: prediction
        """
        self.data = np.genfromtxt("../data/processed_data.csv", delimiter = ' ')
        X = self.data[:,:-1]
        Y = self.data[:, -1]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
        X, Y, test_size=0.33, random_state=42)
        
        self.clf =  OneVsOneClassifier(SVC(kernel='linear'))
       
        self.SCORE = 0
        self.pred_y = 0
        
    def __str__(self):
        return "the score for ONeVSOneSVM is: " + str(self.score)
        
    def draw_confusion(self):
        """
        print confusion matrix: x-axis is prediction, y_axis is ground truth
        """
        #label = np.arange(11,-1,-1)
        label = np.arange(12)
        cnf_matrix = confusion_matrix(self.test_y, self.pred_y,labels = label )
        np.set_printoptions(precision=2)
        
        print cnf_matrix
        
    def run(self):
        """
        fit the model, print confusion matrix and score
        """
        
        self.clf.fit(self.train_x, self.train_y)
        self.score= self.clf.score(self.test_x, self.test_y)
        self.pred_y =self.clf.predict(self.test_x)
        self.draw_confusion()
        print "the score for ONeVSOneSVM is: " + str(self.score)
    
  
        
ovo = OneVSOneSVM()
ovo.run() 
