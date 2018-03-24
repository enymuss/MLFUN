# SVM Average: 0.83%, NN: 84%, Random Forest: 0.95%, NaiveBayes: 0.82%

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import time

def SVMClassifier(data, target, splits=10):
    start_time = time.time()
    nSplits = splits
    kf = KFold(n_splits = nSplits, shuffle=True)
    sum = 0
    for trainIndex, testIndex in kf.split(data):
        xTrain = data[trainIndex]
        yTrain = target[trainIndex].ravel()
        xTest = data[testIndex]
        yTest = target[testIndex]    
        
        clf = svm.SVC()
        clf.fit(xTrain, yTrain)
        sum += clf.score(xTest, yTest)

    avgAccuracy = sum/nSplits
    return avgAccuracy, (time.time() - start_time)

def NN(data, target, splits = 10):
    start_time = time.time()
    nSplits = splits
    kf = KFold(n_splits = nSplits, shuffle=True)
    sum = 0
    for trainIndex, testIndex in kf.split(data):
        xTrain = data[trainIndex]
        yTrain = target[trainIndex].ravel()
        xTest = data[testIndex]
        yTest = target[testIndex]    
        
        clf = MLPClassifier(solver='lbfgs', alpha = 3e-6, hidden_layer_sizes=(10, 2), random_state=1)
        clf.fit(xTrain, yTrain)
        
        sum += clf.score(xTest, yTest)
 
    avgAccuracy = sum/nSplits
    return avgAccuracy, (time.time() - start_time)
    
def RandomForest(data, target, splits = 10):
    start_time = time.time()
    nSplits = splits
    kf = KFold(n_splits = nSplits, shuffle=True)
    sum = 0
    for trainIndex, testIndex in kf.split(data):
        xTrain = data[trainIndex]
        yTrain = target[trainIndex].ravel()
        xTest = data[testIndex]
        yTest = target[testIndex]   
        
        rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
        rf.fit(xTrain, yTrain)
        predicted = rf.predict(xTest)
        accuracy = accuracy_score(yTest, predicted)
        sum += accuracy
        
    avgAccuracy = sum/nSplits
    return avgAccuracy, (time.time() - start_time)

def NaiveBayes(data, target, splits=10):
    start_time = time.time()
    nSplits = splits
    kf = KFold(n_splits = nSplits, shuffle=True)
    sum = 0
    for trainIndex, testIndex in kf.split(data):
        xTrain = data[trainIndex]
        yTrain = target[trainIndex].ravel()
        xTest = data[testIndex]
        yTest = target[testIndex]
        
        gnb = GaussianNB()
        yPred = gnb.fit(xTrain, yTrain).predict(xTest)
        accuracy = accuracy_score(yTest, yPred)
        sum += accuracy
    
    avgAccuracy = sum/nSplits
    return avgAccuracy, (time.time() - start_time)    
         
        
dataArray = np.genfromtxt("spambase/spambase.data", delimiter=',')
data = dataArray[:, 0:57]
target = dataArray[:, 57:58]

cell_text = []

avgAccuracy, duration = SVMClassifier(data, target)
cell_text.append([avgAccuracy, duration])

avgAccuracy, duration = NN(data, target)
cell_text.append([avgAccuracy, duration])

avgAccuracy, duration = RandomForest(data, target)
cell_text.append([avgAccuracy, duration])

avgAccuracy, duration = NaiveBayes(data, target)
cell_text.append([avgAccuracy, duration])

columns = ('Average Accuracy (%)', 'Time (s)')
rows = ('SVM', 'NN', 'Random Forest', 'Naive Bayes')

cell_text = [['%.3f' % j for j in i] for i in cell_text]
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')

plt.subplots_adjust(left=0.25)

plt.show()
