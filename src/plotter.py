import matplotlib,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

LABELS = ["walking-forward", "walking-left", "walking-right", "walking-upstairs", "walking-downstairs",
                "running forward", "jumping Up", "sitting", "standing", "sleeping", "elevator-up", "elevator-down"]


def plot(filename,n_classes,y_test,predictions):

    print "Confusion Matrix:"
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print confusion_matrix
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig("../stats/{}_pig3".format(filename))