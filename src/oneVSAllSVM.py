# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:22:11 2017

@author: Jian Yang local
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class OneVSAllSVM():
    """
    One vs All Ensemble classifier with SVC as core
    """
   
    
    def __init__(self):
        """
        Initialization with following parameters:
            data: data matrix
            
            score: validation accuracy
            pred_y: prediction
        """
#        self.data = np.genfromtxt("../data/processed_data.csv", delimiter = ' ')
        self.data = np.load("../data/stats_data.npy")
        X = self.data[:,:-1]
        Y = self.data[:, -1]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
        X, Y, test_size=0.33, random_state=42)
        
        self.clf =  OneVsRestClassifier(SVC(kernel='linear'))
       
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
        
        class_names = ["walking-forward","walking-left","walking-right","walking-upstairs","walking-downstairs","run","jump","sitting","standing","sleeping","elevator-up","elevator-down"]

        self.plot_confusion_matrix(cnf_matrix, class_names, title='Confusion matrix, without normalization' )
        
        
    def plot_confusion_matrix(self,cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        np.set_printoptions(precision=2)
        plt.figure(figsize=(8, 8))
        
            
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, np.round(cm[i, j],2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
#        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
            
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
    def run(self):
        """
        fit the model, print confusion matrix and score
        """
        
        self.clf.fit(self.train_x, self.train_y)
        self.score= self.clf.score(self.test_x, self.test_y)
        self.pred_y =self.clf.predict(self.test_x)
        self.draw_confusion()
        print "the score for ONeVSOneSVM is: " + str(self.score)
    
  
        
ovo = OneVSAllSVM()
ovo.run() 
