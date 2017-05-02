# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 21:18:20 2017

@author: Jian Yang local
"""

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


from sklearn.metrics import accuracy_score

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:09:51 2017

@author: Jian Yang local
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix




class RandomForest():
    """
    One vs One Ensemble classifier with C as core
    """
    
    def __init__(self):
        """
        Initialization with following parameters:
            data: data matrix
            
            score: validation accuracy
            pred_y: prediction
            
        """
        
#        self.data = np.genfromtxt("../data/processed_data.csv", delimiter = ',')
        self.data = np.genfromtxt("../data/processed_data_window.csv", delimiter = ',')
        X = self.data[:,:-1]
        Y = self.data[:, -1]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
        X, Y, test_size=0.33, random_state=42)
        
        self.clf = RandomForestClassifier(random_state=30 )
        self.SCORE = 0
        self.pred_y = 0
        self.paragrid = {"n_estimators": [10, 20,25, 30, 35]}
        
    def __str__(self):
        return "the score for randomforest is: " + str(self.score)
        
    def draw_confusion(self):
        """
        print confusion matrix: x-axis is prediction, y_axis is ground truth
        """
        #label = np.arange(11,-1,-1)
        label = np.arange(12)
        cnf_matrix = confusion_matrix(self.test_y, self.pred_y,labels = label )
        np.set_printoptions(precision=2)
        
        class_names =["walking-forward","walking-left","walking-right","walking-upstairs","walking-downstairs","running","jumping","sitting","standing","sleeping","elevator-up","elevator-down"]
        
#        print cnf_matrix
        
        self.plot_confusion_matrix(cnf_matrix, class_names, title='Confusion matrix, without normalization' )
        
        
    def plot_confusion_matrix(self,cm, classes,
                          normalize=False,
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
        
#        self.clf.fit(self.train_x, self.train_y)
#        self.score= self.clf.score(self.test_x, self.test_y)
#        self.pred_y =self.clf.predict(self.test_x)
#        self.draw_confusion()
#        print "the score for ONeVSOneSVM is: " + str(self.score)
        
         #grid_search with cross validation   
        score = "neg_mean_squared_error"                
        estimator = GridSearchCV(self.clf, cv=5,
                           param_grid=self.paragrid, scoring =score)
        estimator.fit (self.train_x, self.train_y )
        print("Best parameters set found on development set:")
        print()
        print(estimator.best_params_)
        print()
        print("Grid scores on development set:")
        print()
       
        
        means = -estimator.cv_results_['mean_test_score']
        stds = estimator.cv_results_['std_test_score']
        
        for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
            
            print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std*2 , params))
        print()
            
            
        print("Detailed regression report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
       
        tr_err = np.sqrt(-estimator.cv_results_["mean_train_score"])
        ts_err = np.sqrt(means)
       
       
        self.pred_y =estimator.predict(self.test_x)
        
        self.plot_error_chart("n_estimator", self.paragrid.get("n_estimators"), tr_err, ts_err)
        
#        self.score= estimator.score(self.test_x, self.test_y)
        print  accuracy_score(self.test_y, self.pred_y)

        
        self.draw_confusion()
        
    def plot_error_chart(self,hypername, hyperP_list, train_err, test_err):
   
        plt.plot(hyperP_list, train_err, 'r-', linewidth=4)
        plt.plot(hyperP_list, test_err, 'b-', linewidth=4)
        plt.grid(True)
        plt.xlabel(hypername)
        plt.ylabel("Error Rate")
        plt.title("Train-Test Error Curves "+ hypername)
        leg = plt.legend(["Train Err","Test Err"]);
        leg.get_frame().set_alpha(0.5)
        plt.show()
    
  
        
rf = RandomForest()
rf.run()

