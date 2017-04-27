import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class Hierarchical_model():

    def __init__(self,classifier):
        self.classifier = classifier
        self.data = np.load('../data/binary_data.npy')

    def show_plain_score(self):
        train_x = self.data[:, :-2]
        train_y = self.data[:, -1]
        scores  = cross_val_score(self.classifier, train_x, train_y, cv=5, n_jobs=-1)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def show_hierarchy_score(self):
        train_x = self.data[:, :-1]
        train_y = self.data[:, -1]
        scores = cross_val_score(self.classifier, train_x, train_y, cv=5, n_jobs=-1)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


one_vs_all_svm  = Hierarchical_model(OneVsRestClassifier(SVC()))
one_vs_one_svm  = Hierarchical_model(OneVsOneClassifier(SVC()))
nb_cls = Hierarchical_model(RandomForestClassifier())
rf_cls = Hierarchical_model(RandomForestClassifier())

one_vs_all_svm.show_plain_score()
one_vs_all_svm.show_hierarchy_score()

one_vs_one_svm.show_plain_score()
one_vs_one_svm.show_hierarchy_score()

nb_cls.show_plain_score()
nb_cls.show_hierarchy_score()

rf_cls.show_plain_score()
rf_cls.show_hierarchy_score()