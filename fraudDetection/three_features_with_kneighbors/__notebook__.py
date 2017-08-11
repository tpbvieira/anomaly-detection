# coding: utf-8
# In[1]:
from __future__ import division

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:
# Checking data
print('\n# Checking data')
#print(check_output(["ls", "/media/thiago/ubuntu/datasets/fraudDetection/"]).decode("utf8"))
df = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/Synthetic_Financial_Datasets_For_Fraud_Detection.csv')
#df.head()


# In[3]:
# Counting data
print('\n# Counting data')
from collections import Counter
F = df['type']
print('Total {}'.format(Counter(F)))
F = df[df['type'] == 'PAYMENT']['isFraud']
print('PAYMENT {}'.format(Counter(F)))
F = df[df['type'] == 'TRANSFER']['isFraud']
print('TRANSFER {}'.format(Counter(F)))
F = df[df['type'] == 'CASH_OUT']['isFraud']
print('CASH_OUT {}'.format(Counter(F)))
F = df[df['type'] == 'DEBIT']['isFraud']
print('DEBIT {}'.format(Counter(F)))
F = df[df['type'] == 'CASH_IN']['isFraud']
print('CASH_IN {}'.format(Counter(F)))


# In[4]:
# Metrics Evaluation for random, allZero and allOne scores 
print('\n# Metrics Evaluation for random, allZero and allOne scores ')
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from numpy.random import *
y_true = df['isFraud']
y_scoresR = np.random.randint(0, 2, df.shape[0])
y_scores1 = np.ones(df.shape[0])
y_scores0 = np.zeros(df.shape[0])
print(('Random prediction: Accuracy {}'.format(accuracy_score(y_true, y_scoresR))), ('AUC Score {}'.format(roc_auc_score(y_true, y_scoresR))))
print(('Predict all as Fraud: Accuracy {}'.format(accuracy_score(y_true, y_scores1))), ('AUC Score {}'.format(roc_auc_score(y_true, y_scores1))))
print(('Predict all as Not Fraud: Accuracy {}'.format(accuracy_score(y_true, y_scores0))), ('AUC Score {}'.format(roc_auc_score(y_true, y_scores0))))


# TRANSFER and CASH_OUT will be classified separately


# In[5]:
# Data modeling, feature selection and data transformation
print('\n# [TRANSFER] Data modeling, feature selection and data transformation')
warnings.filterwarnings("ignore", category=FutureWarning)
df_TRANSFER = df[df['type'] ==  'TRANSFER']
X_TRANSFER = np.array(pd.DataFrame(df_TRANSFER, columns=['amount','oldbalanceOrg', 'oldbalanceDest']))
y_TRANSFER = df_TRANSFER['isFraud']
y_TRANSFER = np.array(y_TRANSFER.reshape(len(y_TRANSFER), ))

from sklearn.preprocessing import StandardScaler
sc_TRANSFER = StandardScaler()
sc_TRANSFER.fit(X_TRANSFER)
X_TRANSFER_sc = sc_TRANSFER.transform(X_TRANSFER)


# In[6]:
# data processing for imbalanced data
print('\n# [TRANSFER] data processing for imbalanced data')
from imblearn.over_sampling import SMOTE 
print('Original dataset shape {}'.format(Counter(y_TRANSFER)))
sm = SMOTE(random_state=42)
X_TRANSFER_sm, y__TRANSFER_sm = sm.fit_sample(X_TRANSFER_sc, y_TRANSFER)
print('Resampled dataset shape {}'.format(Counter(y__TRANSFER_sm)))


# In[7]:
# data spliting
print('\n# [TRANSFER] data spliting')
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.cross_validation import train_test_split
warnings.filterwarnings("ignore", category=DeprecationWarning)
X_train, X_test, y_train, y_test = train_test_split(X_TRANSFER_sm, y__TRANSFER_sm, test_size=0.7, random_state=0)


# In[8]:
# KNeighborsClassifier
print('\n# [TRANSFER] KNeighborsClassifier')
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=DeprecationWarning)
scores = ['roc_auc'] # scores = ['roc_auc', 'accuracy'] < please use if you want to run with 'accuracy' basis too.
k = np.arange(10) + 40
tuned_parameters1 = {'n_neighbors': k}
knn = sklearn.neighbors.KNeighborsClassifier()
for score in scores:
	print('\n' + '='*50)
	print(score)
	print('='*50)

	clf1 = GridSearchCV(knn, tuned_parameters1, cv=5, scoring=score, n_jobs=-1)
	clf1.fit(X_train, y_train)

	print ("\n+ best parameters :\n")
	print (clf1.best_estimator_)

	print("\n+ Average score with Training data :\n")
	for params, mean_score, all_scores in clf1.grid_scores_:
		print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

	print("\n+ Reference:\n")
	y_true1, y_pred1 = y_test, clf1.predict(X_test)
	print(classification_report(y_true1, y_pred1))


# In[9]:
# Data modeling, feature selection and data transformation
print('\n# [CASH_OUT] Data modeling, feature selection and data transformation')
df_CASH_OUT = df[df['type'] == 'CASH_OUT']
X_CASH_OUT = np.array(pd.DataFrame(df_CASH_OUT, columns=['amount','oldbalanceOrg', 'oldbalanceDest']))
y_CASH_OUT = df_CASH_OUT['isFraud']
y_CASH_OUT = np.array(y_CASH_OUT.reshape(len(y_CASH_OUT), ))
sc_CASH_OUT = StandardScaler()
sc_CASH_OUT.fit(X_CASH_OUT)
X_CASH_OUT_sc = sc_CASH_OUT.transform(X_CASH_OUT)


# In[10]:
# data processing for imbalanced data
print('\n# [CASH_OUT] data processing for imbalanced data')
print('Original dataset shape {}'.format(Counter(y_CASH_OUT)))
sm = SMOTE(random_state=41)
X_CASH_OUT_sm, y__CASH_OUT_sm = sm.fit_sample(X_CASH_OUT_sc, y_CASH_OUT)
print('Resampled dataset shape {}'.format(Counter(y__CASH_OUT_sm)))


# In[11]:
# data spliting
print('\n# [CASH_OUT] data spliting')
X_train, X_test, y_train, y_test = train_test_split(X_CASH_OUT_sm, y__CASH_OUT_sm, test_size=0.85, random_state=0)


# In[12]:
# KNeighborsClassifier
print('\n# [CASH_OUT] KNeighborsClassifier')
import sklearn
scores = ['roc_auc'] # scores = ['roc_auc', 'accuracy'] < please use if you want to run with 'accuracy' basis too.
k = np.arange(20)+40
tuned_parameters1 = {'n_neighbors': k}
knn = sklearn.neighbors.KNeighborsClassifier()
for score in scores:
	print('\n' + '='*50)
	print(score)
	print('='*50)

	clf2 = GridSearchCV(knn, tuned_parameters1, cv=5, scoring=score, n_jobs=-1)
	clf2.fit(X_train, y_train)

	print ("\n+ best parameters :\n")
	print (clf2.best_estimator_)

	print("\n+ Average score with Training data :\n")
	for params, mean_score, all_scores in clf2.grid_scores_:
		print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))
	
	print("\n+ Reference:\n")
	y_true2, y_pred2 = y_test, clf2.predict(X_test)
	print(classification_report(y_true2, y_pred2))
