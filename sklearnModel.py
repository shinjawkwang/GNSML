import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
#rfModel = RandomForestRegressor(n_estimators=100)
#svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
svm = SVC()

# Deleted log1p, because there are too many labels
# we cannot cover when checking precision, recall, f_score
# yLabelsLog = np.log(yLabels+3)


#rfModel.fit(dataTrain, yLabels.values.ravel())

svm.fit(dataTrain, yLabels.values.ravel())

neigh = KNeighborsClassifier()
neigh.fit(dataTrain, yLabels.values.ravel())



# Test Trained Random Forest Regressor
#preds = rfModel.predict(X=dataTest)
print("3")
svm_pred = svm.predict(X=dataTest)
neigh_pred = neigh.predict(X=dataTest)
print("4")
testYLabels = testYLabels.values.ravel()
# testLog = np.log(testYLabels+3)
# testLog = testLog.values.ravel()


# evaluation values (or matrix)
#aScore = accuracy_score(testYLabels, preds.round())
#cMatrix = confusion_matrix(testYLabels, preds.round())

print("6")
svm_cMatrix = confusion_matrix(testYLabels, svm_pred)
neigh_cMatrix = confusion_matrix(testYLabels, neigh_pred)
print("7")

# ignore '0' value for displaying
'''cMatrixDP = np.delete(cMatrix, 2, 0)
cMatrixDP = np.delete(cMatrixDP, 2, 1)'''

svm_cMatrixDP = np.delete(svm_cMatrix, 2, 0)
neigh_cMatrixDP = np.delete(neigh_cMatrix, 2, 0)
print("8")
svm_cMatrixDP = np.delete(svm_cMatrixDP, 2, 1)
neigh_cMatrixDP = np.delete(neigh_cMatrixDP, 2, 1)
'''
precisionList = precision(cMatrix)
recallList = recall(cMatrix)
precisionList = np.delete(precisionList, 2)
recallList = np.delete(recallList, 2)
f1 = f_score(1, precisionList, recallList)'''

# plot confusion matrix
'''
plt.figure()
plot_confusion_matrix(cMatrixDP, classes, normalize=True, title='Confusion Matrix With Normalization')
plt.figure()
plot_confusion_matrix(cMatrixDP, classes, normalize=False, title='Confusion Matrix Without Normalization')'''
plt.figure()
plot_confusion_matrix(svm_cMatrixDP, classes, normalize=True, title='Confusion svm_Matrix With Normalization')
plt.figure()
plot_confusion_matrix(svm_cMatrixDP, classes, normalize=False, title='Confusion svm_Matrix Without Normalization')
plt.figure()
plot_confusion_matrix(neigh_cMatrixDP, classes, normalize=True, title='Confusion neigh_Matrix With Normalization')
plt.figure()
plot_confusion_matrix(neigh_cMatrixDP, classes, normalize=False, title='Confusion neigh_Matrix Without Normalization')
plt.show()

print("5")
svm_aScore = accuracy_score(testYLabels, svm_pred.round())
neigh_aScore = accuracy_score(testYLabels, neigh_pred.round())
print("9")
svm_precisionList = precision(svm_cMatrix)
neigh_precisionList = precision(neigh_cMatrix)
print("10")
svm_recallList = recall(svm_cMatrix)
neigh_recallList = recall(neigh_cMatrix)
print("11")
svm_precisionList = np.delete(svm_precisionList, 2)
neigh_precisionList = np.delete(neigh_precisionList, 2)
print("12")
svm_recallList = np.delete(svm_recallList, 2)
neigh_recallList = np.delete(neigh_recallList, 2)
print("13")
svm_f1 = f_score(1, svm_precisionList, svm_recallList)
neigh_f1 = f_score(1, neigh_precisionList, neigh_recallList)
print("14")



# print result
print("knn")
print("Accuracy:", neigh_aScore)
print("Precision:", neigh_precisionList)
print("Recall:", neigh_recallList)
print("F1 Score:", neigh_f1)

print("svm")
print("Accuracy:", svm_aScore)
print("Precision:", svm_precisionList)
print("Recall:", svm_recallList)
print("F1 Score:", svm_f1)