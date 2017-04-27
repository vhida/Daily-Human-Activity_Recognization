import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class Hierarchical_model():

    def __init__(self,classifier):
        self.classifier = classifier
        self.data = np.load('../data/binary_data.npy')

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
            kf = KFold(n_splits=cv,shuffle=True)
            for train_index, test_index in kf.split(X):
                X_train,X_test =  X[train_index],X[test_index]
                y_train,y_test =  y[train_index],y[test_index]
                y_test =  y[test_index]

            model = self.classifier()
            model.fit(X_train,y_train)
            accuracy = accuracy_score(y_test,model.predict(X_test))
            acc.append(accuracy)
        return scores(np.asarray(acc))


svc  = Hierarchical_model(SVC)
nb_cls = Hierarchical_model(RandomForestClassifier)
rf_cls = Hierarchical_model(RandomForestClassifier)

svc.show_plain_score(6)
svc.show_hierarchy_score(6)

nb_cls.show_plain_score(6)
nb_cls.show_hierarchy_score(6)

rf_cls.show_plain_score(6)
rf_cls.show_hierarchy_score(6)