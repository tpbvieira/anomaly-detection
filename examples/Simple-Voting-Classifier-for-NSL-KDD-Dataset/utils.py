import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats

class Processor:
	# preprocess the data
	@staticmethod
	def process(fid, normalization='zscore'):
		if fid == None:
			return
		content = open(fid,'r').read().splitlines()
		mat = np.empty((len(content),41),dtype='S10')
		y = []
		for i in xrange(len(content)):
			splited = content[i].split(',')
			mat[i, :] = splited[:-2]
			y.append(int(splited[-1]))
		# for data X
		dataIdx = np.arange(4,41)
		X = mat[:,dataIdx].astype(np.float)
		# for categorical features at 0-3
		for i in xrange(3,-1,-1):
			temp = pd.Categorical(pd.Series(mat[:,i]))
			# insert colomn to the front
			X = np.insert(X,0,temp.codes, axis=1)
		# normalization: l1 or l2, z-score
		if normalization == 'zscore':
			X = stats.zscore(X, axis=1, ddof=1)
		elif normalization == 'l1':
			X = preprocessing.normalize(X,norm='l1',axis=1)
		elif normalization == 'l2':
			X = preprocessing.normalize(X,norm='l2',axis=1)
		else:
			raise ValueError('Wrong normalization type')
		# index abnormal instances
		abnId = [i for i in range(len(y)) if y[i]!=21]
		return (X,np.array(y),np.array(abnId))
	# compare classifiers 
	@staticmethod
	def compareClfs(clfs, trainX, trainY, testX, testY):
		# clfs is a dictionary of classifiers
		result = {}; bestErr = 1; bestName = ''
		for name,clf in clfs.iteritems():
			try:
				predictY = clf.best_estimator_.fit(trainX, trainY).predict(testX)
			except:
				predictY = clf.fit(trainX, trainY).predict(testX)
			result[name] = 1-np.mean(predictY==testY)
			print name,':',result[name]
			if result[name] < bestErr:
				bestErr = result[name]
				bestName = name
		print "Best classifier",name,':',bestErr
		return result

class EnsembleClassifier:
	def __init__(self, classifiers=None):
		self.classifiers = classifiers
	def fit(self, X, y):
		for classifier in self.classifiers:
			classifier.fit(X, y)
	def predict(self, X):
		self.predictions_ = list()
		for classifier in self.classifiers:
			try:
				self.predictions_.append(classifier.best_estimator_.predict(X))
			except:
				self.predictions_.append(classifier.predict(X))
		# vote the results
		predMat = np.array(self.predictions_)
		result = np.zeros(predMat.shape[1])
		for i in xrange(predMat.shape[1]):
			temp = np.bincount(predMat[:,i])
			# if all disagree with each other, vote for the first classifier
			if np.max(temp) == 1:
				result[i] = predMat[0,i]
			else:
				result[i] = np.argmax(temp)
		return result