import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Hierarchical_model():

    def __init__(self,classifier,filename):
        self.classifier = classifier
        self.data = np.load(filename)

    def show_plain_score(self,cv):
        train_x = self.data[:, :-2]
        train_y = self.data[:, -1]
        scores = self.cv_func(cv,train_x,train_y)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def show_hierarchy_score(self,cv):
        train_x = self.data[:, :-1]
        train_y = self.data[:, -1]
        scores = self.cv_func(cv,train_x,train_y)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def cv_func(self,cv,X,y,mode = None):
        """
        :param: cv

        :return:avergae accuracy rate
        """
        acc = []
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
            X_train,X_test, y_train,y_test =  train_test_split(X,y,test_size=0.3,random_state=50)

            model = self.classifier()
            model.fit(X_train,y_train)
            accuracy = accuracy_score(y_test,model.predict(X_test))
            acc.append(accuracy)
        return scores(np.asarray(acc))

#filename = '../data/binary_data.npy'
filename = '../data/condensed_time_series_data.npy'
svc  = Hierarchical_model(SVC,filename)
nb_cls = Hierarchical_model(GaussianNB,filename)
rf_cls = Hierarchical_model(RandomForestClassifier,filename)

svc.show_plain_score(6)
svc.show_hierarchy_score(6)

nb_cls.show_plain_score(6)
nb_cls.show_hierarchy_score(6)

rf_cls.show_plain_score(6)
rf_cls.show_hierarchy_score(6)