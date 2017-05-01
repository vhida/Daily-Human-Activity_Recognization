import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import plotter
from sklearn import metrics
from sklearn.model_selection import train_test_split



class Cluster_classifier(BaseEstimator,ClassifierMixin):


    def __init__(self, K, classifier):
        '''
        Create a cluster classifier object
        '''
        self.K = K
        self.classifier = classifier
        self.model = KMeans(n_clusters=self.K, random_state=0)
        self.clss = []

    def clustering_accuracy(self,X,Y):
        self.model.fit(X, Y)
        labels = self.model.labels_
        for i in set(labels):
            indice = np.where(labels == i)[0]
            real_labels = Y[indice]
            print (np.unique(real_labels),np.std(real_labels))


    def fit(self, X, Y):
        '''
        Learn a cluster classifier object
        '''
        self.model.fit(X)
        labels = self.model.labels_
        # get the classifier
        # Train the model for each cluster
        for c_no in range(self.K):
            # print "K is ", c_no, "-----------------------------------------------------"
            w_c_X = X
            w_c_Y = Y
            list = []
            for index in range(len(labels)):
                # collect data cases in the cluster
                if c_no == labels[index]:
                    list.append(index)
            # print "index for ", c_no
            #print list
            w_c_X = w_c_X[list]
            w_c_Y = w_c_Y[list]
            # print "number in cluster ", c_no, " is ", w_c_Y.shape, "and unique size ::", np.unique(w_c_Y).size
            # print w_c_Y
            if w_c_Y.size != 0:
                if np.unique(w_c_Y).size == 1:
                    self.clss.append(("no_fit", w_c_Y[0]))
                else:
                    # print "beging fitting-------------------------"
                    c_m = self.classifier()
                    c_m.fit(w_c_X, w_c_Y)
                    # print w_c_Y.shape, w_c_X.shape
                    self.clss.append(c_m)
            else:
                # if no data case assigned to that cluster, use the untrained model to give random classification
                self.clss.append(self.classifier() )

    def predict(self, X):
        '''
        Make predictions usins a cluster classifier object
        '''
        pred_clazz = np.arange(X.shape[0])
        labels = self.model.predict(X)
        # use within cluster classifier to get the predicted class
        # predict each data case using the model trained on the cluster it belongs to
        for index in np.arange(labels.shape[0]):
            predictor = self.clss[labels[index]]
            if isinstance(predictor,tuple):
                pred_clazz[index]=predictor[1]
            else:
                pred_clazz[index] = predictor.predict(X[index].reshape(1, -1))
        return pred_clazz

    def score(self, X, test_y):

        return accuracy_score(test_y, self.predict(X))

    def cv_func(self,cv,X,y):
        """
        :param: cv

        :return:avergae accuracy rate
        """
        acc = []
        preds = []
        class scores():
            def __init__(self,arr):
                self.arr = arr

            def mean(self):
                return np.mean(self.arr)

            def std(self):
                return np.std(self.arr)

            def get(self):
                return self.arr

        for i in range(cv):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.fit(X_train,y_train)
            pred = self.predict(X_test)
            accuracy = accuracy_score(y_test,pred)
            acc.append(accuracy)
            preds.append([y_test,pred])
        return scores(np.asarray(acc)),np.asarray(preds)[-1]

#data = np.load('../data/binary_data.npy')
data = np.load('../data/condensed_time_series_data.npy')

train_x = data[:, :-2]
train_y = data[:, -1]

cls = Cluster_classifier(12,SVC)
scores,preds = cls.cv_func(10,train_x,train_y)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
print  metrics.confusion_matrix(preds[0], preds[1])
plotter.plot("../stats/___svm_clustering",12,preds[0],preds[1])
print scores.get()

cls = Cluster_classifier(12,RandomForestClassifier)
scores,preds = cls.cv_func(10,train_x,train_y)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
plotter.plot("../stats/___rf_clustering",12,preds[0],preds[1])
print scores.get()

cls = Cluster_classifier(12,GaussianNB)
scores,preds = cls.cv_func(10,train_x,train_y)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
plotter.plot("../stats/___nb_clustering",12,preds[0],preds[1])
print scores.get()
