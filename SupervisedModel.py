import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix


# names of features; change when data has been changed
classes = ['Unknown', 'Abnormal', 'Normal']


# confusion matrix plotting
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# rmsle method
def rmsle(y_test, y_pred):
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# calculate recall method
def recall(cmatrix):
    recall_list = []
    cnt = 0
    for row in cmatrix:
        sum = 0
        for col in row:
            sum += col
        recall_list.append(row[cnt]/sum)
        cnt += 1
    return recall_list


# calculate precision method
def precision(cmatrix):
    precision_list = []
    for col in range(cmatrix.shape[1]):
        sum = 0
        for row in range(cmatrix.shape[0]):
            sum += cmatrix[row][col]
        precision_list.append(cmatrix[col][col]/sum)
    return precision_list


# calculate f score method
def f_score(beta, precision, recall):
    recall = np.array(recall)
    precision = np.array(precision)
    return ((1+beta**2) * (recall * precision)) / ((beta**2)*precision + recall)


# read data
dataTrain = pd.read_csv('dataset/traindat.txt')
dataTest = pd.read_csv('dataset/testdat.txt')
yLabels = pd.read_csv('dataset/traintar.txt')
testYLabels = pd.read_csv('dataset/testtar.txt')


# Data Training
rfModel = RandomForestRegressor(n_estimators=100)


# Deleted log1p, because there are too many labels
# we cannot cover when checking precision, recall, f_score
# yLabelsLog = np.log(yLabels+3)


rfModel.fit(dataTrain, yLabels.values.ravel())


# Test Trained Random Forest Regressor
preds = rfModel.predict(X=dataTest)

testYLabels = testYLabels.values.ravel()
# testLog = np.log(testYLabels+3)
# testLog = testLog.values.ravel()


# evaluation values (or matrix)
aScore = accuracy_score(testYLabels, preds.round())
cMatrix = confusion_matrix(testYLabels, preds.round())

# ignore '0' value for displaying
# cMatrixDP = np.delete(cMatrix, 2, 0)
# cMatrixDP = np.delete(cMatrixDP, 2, 1)

precisionList = precision(cMatrix)
recallList = recall(cMatrix)
precisionList = np.delete(precisionList, 2)
recallList = np.delete(recallList, 2)
f1 = f_score(1, precisionList, recallList)


# plot confusion matrix
plt.figure()
plot_confusion_matrix(cMatrixDP, classes, normalize=True, title='Confusion Matrix With Normalization')
plt.figure()
plot_confusion_matrix(cMatrixDP, classes, normalize=False, title='Confusion Matrix Without Normalization')
plt.show()


# print result
print("Accuracy:", aScore)
print("Precision:", precisionList)
print("Recall:", recallList)
print("F1 Score:", f1)
