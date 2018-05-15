# coding:utf-8
## Recursive Feature Elimination to identify the f features more importants for iGovTI classification
from sklearn import preprocessing
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from numpy import genfromtxt

## settings
csf = 54
success = 0.0;
fcs_success = 0.0;

## loading data
print('\n# loading data')
data = genfromtxt('data/class_data_raw', delimiter='\t')
target = genfromtxt('data/class_target', delimiter='\t')
qualitative_fcs = genfromtxt('data/class_fcs', delimiter='\t', dtype=None)

## create a base classifier
svm = SVC(kernel='linear')

## create the RFE model and select csf features, which are the critical success factors
rfe = RFE(svm, csf)
rfe = rfe.fit(data, target)

## get results
for i in range(len(rfe.support_)):
	if rfe.support_[i] == qualitative_fcs[i]:
		success = success + 1
		if (rfe.support_[i] == True):
			fcs_success = fcs_success + 1

## success ratio
print('\n# success ratio')
print success/len(rfe.support_)
print fcs_success/54

## success ratio
print('\n# support')
print rfe.support_

## success ratio
print('\n# ranking')
print rfe.ranking_