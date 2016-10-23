from sklearn import svm,cross_validation,tree,linear_model,preprocessing,metrics
from sklearn.mixture import GMM
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from utils import Processor, EnsembleClassifier
import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir('/Users/Wei/Desktop/nslkdd')
trainFid = 'KDDTrain+.txt'
testFid = 'KDDTest+.txt'

(trainX,trainY,trainAbnId) = Processor.process(trainFid,'l2')
(testX,testY,testAbnId) = Processor.process(testFid,'l2')

# prepare data
# layer 1: normal v.s. abnormal
trainX1 = trainX
trainY1 = (trainY==21).astype(int)
testX1 = testX
testY1 = (testY==21).astype(int)
# layer 2: among abnormal
trainX2 = trainX[trainAbnId,:]
trainY2 = trainY[trainAbnId]

########################### 1st layer ############################
print "First layer:"
lrParas = {'penalty':['l1','l2'],'C':np.linspace(0.001,1.5,5)}
svmParas = {'kernel':'rbf','C':np.linspace(0.001,1.5,5),'gamma':np.linspace(0.001,1.5,5)}
ks = {'n_neighbors':[3,5]}
clfs1 = dict(
			dt = DecisionTreeClassifier(),\
			rf = RandomForestClassifier(),\
			bnb = BernoulliNB(),\
			lr = GridSearchCV(LogisticRegression(), lrParas),\
			adb = AdaBoostClassifier(),\
			knn = GridSearchCV(KNeighborsClassifier(),ks),\
			grid = GridSearchCV(svm.SVC(), svmParas)
			)
result = Processor.compareClfs(clfs1, trainX1, trainY1, testX1, testY1)
print result

# ########################### 2nd layer ############################
ks = {'n_neighbors':[3,5,7,9]}
lrParas = {'penalty':['l1','l2'],'C':np.linspace(0.001,1.5,5)}
clfs2 = [RandomForestClassifier(),\
		tree.DecisionTreeClassifier(),\
		BernoulliNB(),\
		AdaBoostClassifier(),\
		GridSearchCV(KNeighborsClassifier(),ks),\
		GridSearchCV(LogisticRegression(), lrParas)
		]
ec = EnsembleClassifier(clfs2)
ec.fit(trainX2, trainY2)
result = ec.predict(testX2)
print 'err2:',1-np.mean(result==testY2)
err2

predY = np.zeros(len(trainY))
predNId = [i for i in range(len(predY1)) if predY1[i] == 1]
predY[predNId] = predY1
predY[predAbnId] = result
print 'err:',1-np.mean(testY==predY)