# coding: utf-8
# ## EDA and Fraud detection
# 
# The provided data has the financial transaction data as well as the target variable **isFraud**, which is the actual
# fraud status of the transaction and **isFlaggedFraud** is the indicator which the simulation is used to flag the
# transaction using some threshold. The goal should be how we can improve and come up with better threshold to capture
# the fraud transaction.
#
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division
import os.path
import warnings
import time
import datetime
import numpy as np  																									# linear algebra
import pandas as pd  																									# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.stats import skew, boxcox
from statsmodels.tools import categorical
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLasso
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, auc, \
	roc_auc_score, roc_curve, classification_report
from sklearn.manifold import TSNE
from print_feature_ranking import print_feature_ranking

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
sns.set_style("dark")


# return string dateTime
def now_datetime_str():
	tmp_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	return tmp_time


# Read CSV file into a Panda DataFrame and print some information
def read_csv(csv):
	df = pd.read_csv(csv)
	print("{}: {} has {} observations and {} columns".format(now_datetime_str(), csv, df.shape[0], df.shape[1]))
	print("{}: Column name checking::: {}".format(now_datetime_str(), df.columns.tolist()))
	return df


# function to read dataframe and find the missing data on the columns and # of missing
def checking_missing(df):
	try:
		if isinstance(df, pd.DataFrame):
			df_na_bool = pd.concat([df.isnull().any(), df.isnull().sum(), (df.isnull().sum()/df.shape[0])*100], axis=1, keys=['df_bool', 'df_amt', 'missing_ratio_percent'])
			df_na_bool = df_na_bool.loc[df_na_bool['df_bool'] == True]
			return df_na_bool
		else:
			print("{}: The input is not panda DataFrame".format(now_datetime_str()))

	except (UnboundLocalError, RuntimeError):
		print("{}: Something is wrong".format(now_datetime_str()))


# Plots a Correlation Heatmap
def plot_correlation_heatmap(dataframe, title):
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)
	cmap = cm.get_cmap('jet', 30)
	cax = ax1.imshow(dataframe.corr(), interpolation="nearest", cmap=cmap)
	ax1.grid(True)
	plt.title(title)
	labels = dataframe.columns.tolist()
	ax1.set_xticklabels(labels, fontsize=13, rotation=45)
	ax1.set_yticklabels(labels, fontsize=13)
	fig.colorbar(cax)
	plt.show()


# Defines plot_confusion_matrix function
def plot_confusion_matrix(_cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.imshow(_cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	if normalize:
		_cm = _cm.astype('float') / _cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(_cm)

	thresh = _cm.max() / 2.
	for i, j in itertools.product(range(_cm.shape[0]), range(_cm.shape[1])):
		plt.text(j, i, _cm[i, j], horizontalalignment="center", color="white" if _cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


# Read CSV file into a Panda DataFrame and print some information
print("\n## Loading data")
raw_data = read_csv("/media/thiago/ubuntu/datasets/fraudDetection/Synthetic_Financial_Datasets_For_Fraud_Detection.csv")


# ### 1. EDA (exploratory data analysis ) ##############################################################################
# In this section, we will do EDA to understand the data more. From the simulation, there are 5 transaction types as per
#  illustrated below.


# # Check if there's any null values.
# print("\n## Let's check the dataset if there's any null values.")
# print(checking_missing(raw_data))


# # Look at the dataset sample and other properties.
# print("\n## Head:")
# print(raw_data.head(5))
# print("\n## Describe:")
# print(raw_data.describe())
# print("\n## Info:")
# print(raw_data.info())


# # Plot transaction count by transaction type
# print("\n## Plot transaction count by transaction type:")
# f, ax = plt.subplots(1, 1, figsize=(8, 8))
# raw_data.type.value_counts().plot(kind='bar', title="Transaction count by transaction type", ax=ax, figsize=(8,8))
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()


# # *isFraud* is the indicator which indicates the actual fraud transactions
# # *isFlaggedFraud* is what the system prevents the transaction due to some thresholds being triggered.
# # Plot Fraud (1) and Legitmate (0) transactions count by transaction type
# print("\n## Plot Fraud (1) and Legitmate (0) transactions count by transaction type:")
# ax = raw_data.groupby(['type', 'isFraud']).size().plot(kind='bar')
# ax.set_title("Fraud (1) and Legitmate (0) transactions count by transaction type")
# ax.set_xlabel("(Type, isFraud)")
# ax.set_ylabel("Count of transaction")
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()
# # Observation: Only TRANSFER and CASH_OUT transactions have fraud


# # Plot transactions which are flagged as isFlaggedFraud per transaction type
# print("\n## Plot Flagged Fraud (1) and Legitmate (0) transactions count by transaction type:")
# ax = raw_data.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')
# ax.set_title("Flagged Fraud (1) and Legitmate (0) transactions count by transaction type")
# ax.set_xlabel("(Type, isFlaggedFraud)")
# ax.set_ylabel("Count of transaction")
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()
# # Observation: the system can flag only 16 transfer transactions as fraud.


# # Plot fraud *TRANSFER* analysis
# print("\n## Plot fraud *TRANSFER* analysis")
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# fig.suptitle("Analyse *TRANSFER* flagged as fraud", fontsize="x-large")
# tmpData = raw_data.loc[(raw_data.type == 'TRANSFER'), :]
# a = sns.boxplot(x = 'isFraud', y = 'amount', data = tmpData, ax=axs[0][0])
# axs[0][0].set_yscale('log')
# b = sns.boxplot(x = 'isFraud', y = 'oldbalanceDest', data = tmpData, ax=axs[0][1])
# axs[0][1].set(ylim=(0, 0.5e8))
# c = sns.boxplot(x = 'isFraud', y = 'oldbalanceOrg', data=tmpData, ax=axs[1][0])
# axs[1][0].set(ylim=(0, 3e7))
# d = sns.regplot(x = 'oldbalanceOrg', y = 'amount', data=tmpData.loc[(tmpData.isFraud==1), :], ax=axs[1][1])
# plt.show()
# # Plot fraud *CASH_OUT* analysis
# print("\n## Plot fraud *CASH_OUT* analysis")
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# fig.suptitle("Analyse *CASH_OUT* flagged as fraud", fontsize="x-large")
# tmpData = raw_data.loc[(raw_data.type == 'CASH_OUT'), :]
# a = sns.boxplot(x = 'isFraud', y = 'amount', data = tmpData, ax=axs[0][0])
# axs[0][0].set_yscale('log')
# b = sns.boxplot(x = 'isFraud', y = 'oldbalanceDest', data = tmpData, ax=axs[0][1])
# axs[0][1].set(ylim=(0, 0.5e8))
# c = sns.boxplot(x = 'isFraud', y = 'oldbalanceOrg', data=tmpData, ax=axs[1][0])
# axs[1][0].set(ylim=(0, 3e7))
# d = sns.regplot(x = 'oldbalanceOrg', y = 'amount', data=tmpData.loc[(tmpData.isFraud==1), :], ax=axs[1][1])
# plt.show()


# ### 2. Modeling ######################################################################################################


# focus only on **TRANSFER** and **CASH_OUT** (where there are fraud), data slicing and data transformation
print("\n## focus only on **TRANSFER** and **CASH_OUT** (where there are fraud)")
# Keep only interested transaction type ('TRANSFER', 'CASH_OUT')
tmpData = raw_data.loc[(raw_data['type'].isin(['TRANSFER', 'CASH_OUT'])), :]


# Data slicing - Drop unnecessary data ('step', 'nameOrig', 'nameDest', 'isFlaggedFraud')
print("\n## Data slicing - Drop unnecessary data ('step', 'nameOrig', 'nameDest', 'isFlaggedFraud')")
# tmpData.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
tmpData.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
tmpData = tmpData.reset_index(drop=True)
# Convert categorical variables to numeric variable
a = np.array(tmpData['type'])
b = categorical(a, drop=True)
tmpData['type_num'] = b.argmax(1)
tmpData.drop(['type'], axis=1, inplace=True)


# # Plot Correlations of TRANSFER and CASH_OUT transactions and selected features
# print("\n## Plot Correlations of TRANSFER and CASH_OUT transactions and selected features")
# plotCorrelationHeatmap(tmpData, "TRANSFER and CASH_OUT Correlation")
# plotCorrelationHeatmap(raw_data.loc[(raw_data.type == 'TRANSFER'), :], "TRANSFER Correlation")
# plotCorrelationHeatmap(raw_data.loc[(raw_data.type == 'CASH_OUT'), :], "CASH_OUT Correlation")


# # Quickly get the count and the target variable count.
# print("\n## Plot Transaction count by type")
# ax = tmpData.type.value_counts().plot(kind='bar', title="Transaction count by type", figsize=(6,6))
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()
# print("\n## Plot Fraud (1) and Legitmate (0) transactions count")
# ax = pd.value_counts(tmpData['isFraud'], sort = True).sort_index().plot(kind='bar', title="Fraud (1) and Legitmate (0)
#  transactions count")
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))
# plt.show()


# ## 2.1 Feature extraction ############################################################################################
# Based on the dataset, the numeric variables are quite skew, in this case. I will try to scale it with 2 methods
# (SQRT and Box-Cox) and compare them on the graph.
# skewness of the distribution: Normally distributed data has skewness should be about 0. A skewness value > 0 means
# that there is more weight in the left tail of the distribution
# Boxcox transformation: makes the data normal
print("\n## Scale features with SQRT and Box-Cox to compare them on the graph")

# print("\n## Plot Transformations for **amount**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.title("Amount Histogram")
# plt.hist(tmpData['amount'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("Transaction amount")
# plt.text(10,100000,"Skewness: {0:.2f}".format(skew(tmpData['amount'])))
#
# figure.add_subplot(132)
# plt.title("SQRT on amount histogram")
# plt.hist(np.sqrt(tmpData['amount']), facecolor = 'red', alpha=0.5)
# plt.xlabel("Square root of amount")
# plt.text(10, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmpData['amount']))))

tmpData['amount_boxcox'] = preprocessing.scale(boxcox(tmpData['amount']+1)[0])
# figure.add_subplot(133)
# plt.title("Box-cox on amount histogram")
# plt.hist(tmpData['amount_boxcox'], facecolor = 'red', alpha=0.5)
# plt.xlabel("Box cox of amount")
# plt.text(10, 100000, "Skewness: {0:.2f}".format(skew(tmpData['amount_boxcox'])))
# plt.show()
# High skewness on left side but box-cox reveals normal distribution


# print("\n## Plot Transformations for **oldbalanceOrg**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.title("oldbalanceOrg Histogram")
# plt.hist(tmpData['oldbalanceOrg'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("old balance originated")
# plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmpData['oldbalanceOrg'])))
#
# figure.add_subplot(132)
# plt.title("SQRT on oldbalanceOrg histogram")
# plt.hist(np.sqrt(tmpData['oldbalanceOrg']), facecolor = 'red', alpha=0.5)
# plt.xlabel("Square root of oldBal")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmpData['oldbalanceOrg']))))

tmpData['oldbalanceOrg_boxcox'] = preprocessing.scale(boxcox(tmpData['oldbalanceOrg']+1)[0])
# figure.add_subplot(133)
# plt.title("Box-cox on oldbalanceOrg histogram")
# plt.hist(tmpData['oldbalanceOrg_boxcox'], facecolor = 'red', alpha=0.5)
# plt.xlabel("Box cox of oldBal")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmpData['oldbalanceOrg_boxcox'])))
# plt.show()
# # High skewness on left side but box-cox reveals an outlier


# print("\n## Plot Transformations for **newbalanceOrg**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.title("newbalanceOrig histogram")
# plt.hist(tmpData['newbalanceOrig'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("New balance originated")
# plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmpData['newbalanceOrig'])))
#
# figure.add_subplot(132)
# plt.title("SQRT on newbalanceOrig histogram")
# plt.hist(np.sqrt(tmpData['newbalanceOrig']), facecolor = 'red', alpha=0.5)
# plt.xlabel("Square root of newBal")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmpData['newbalanceOrig']))))

tmpData['newbalanceOrg_boxcox'] = preprocessing.scale(boxcox(tmpData['newbalanceOrig']+1)[0])
# figure.add_subplot(133)
# plt.title("Box-cox on newbalanceOrig histogram")
# plt.hist(tmpData['newbalanceOrg_boxcox'], facecolor = 'red', alpha=0.5)
# plt.xlabel("Box cox of newBal")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmpData['newbalanceOrg_boxcox'])))
# plt.show()
# # High skewness on left side, including box-cox reveals an outlier


# print("\n## Plot Transformations for **oldbalanceDest**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.hist(tmpData['oldbalanceDest'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("Old balance desinated")
# plt.title("oldbalanceDest histogram")
# plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmpData['oldbalanceDest'])))
#
# figure.add_subplot(132)
# plt.hist(np.sqrt(tmpData['oldbalanceDest']), facecolor = 'red', alpha=0.5)
# plt.xlabel("Square root of oldBalDest")
# plt.title("SQRT on oldbalanceDest histogram")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmpData['oldbalanceDest']))))

tmpData['oldbalanceDest_boxcox'] = preprocessing.scale(boxcox(tmpData['oldbalanceDest']+1)[0])
# figure.add_subplot(133)
# plt.hist(tmpData['oldbalanceDest_boxcox'], facecolor = 'red', alpha=0.5)
# plt.xlabel("Box cox of oldbalanceDest")
# plt.title("Box cox on oldbalanceDest histogram")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmpData['oldbalanceDest_boxcox'])))
# plt.show()
# # High skewness on left side but box-cox reveals skewness to right side


# print("\n## Plot Transformations for **newbalanceDest**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.hist(tmpData['newbalanceDest'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("newbalanceDest")
# plt.title("newbalanceDest histogram")
# plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmpData['newbalanceDest'])))
#
# figure.add_subplot(132)
# plt.hist(np.sqrt(tmpData['newbalanceDest']), facecolor = 'red', alpha=0.5)
# plt.xlabel("Square root of newbalanceDest")
# plt.title("SQRT on newbalanceDest histogram")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmpData['newbalanceDest']))))

tmpData['newbalanceDest_boxcox'] = preprocessing.scale(boxcox(tmpData['newbalanceDest']+1)[0])
# figure.add_subplot(133)
# plt.hist(tmpData['newbalanceDest_boxcox'], facecolor = 'red', alpha=0.5)
# plt.xlabel("Box cox of newbalanceDest")
# plt.title("Box cox on newbalanceDest histogram")
# plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmpData['newbalanceDest_boxcox'])))
# plt.show()
# # High skewness on left side but box-cox reveals normal distribution


# filtered unrelated transaction type out and keep only relevant.
# In this notebook, I will quickly use traditional *under*-sampling method (there are several other ways; under and
# over sampling, SMOTE, etc).
# Also we will use only the boxcox data transformation for prediction.
print("\n## Feature Selection: Use only the box-cox data transformation for prediction")
# tmpData.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','amount'],axis=1,inplace=True)
# print(tmpData.head(5))
# Plot Correlations of TRANSFER and CASH_OUT transactions and box-cox features
# print("\n## Plot Correlations of TRANSFER and CASH_OUT transactions and box-cox features")
# sns.heatmap(tmpData)
# plt.show()
# sns.heatmap(tmpData.loc[(tmpData.type_num == 0), :].corr())
# plt.show()
# sns.heatmap(tmpData.loc[(tmpData.type_num == 1), :].corr())
# plt.show()

# print("\n## Fraud Ratio
#  (TRANSFER and CASH_OUT):")
# print("% of normal transactions: ", len(tmpData[tmpData.isFraud == 0])/len(tmpData))
# print("% of fraud transactions: ", len(tmpData[tmpData.isFraud == 1])/len(tmpData))
# print("Total number of transactions in resampled data: ", len(tmpData))
# # There're only actual fraud of 0.3% and this is very imbalance data.


# I will under sample the dataset by creating a 50-50 ratio of randomly selecting 'x' amount of sample from majority
# class, with 'x' being the total number of records with the minority class.
# Number of data points in the minority class

# Preparing data for training
print("\n## Preparing data for training...")
# Whole dataset
print("\n## Whole dataset:")
data = tmpData.ix[:, tmpData.columns != 'isFraud']
target = tmpData.ix[:, tmpData.columns == 'isFraud']
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)

if not os.path.isfile('/media/thiago/ubuntu/datasets/fraudDetection/train_data.csv'):
	train_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data.csv', index=True)
	train_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_target.csv', index=True)
	fraud_train_indices = train_target[train_target.isFraud == 1].index.values
	fraud_train_data = train_data.ix[fraud_train_indices, :]
	fraud_train_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_train_data.csv', index=True)
	fraud_train_target = train_target.ix[fraud_train_indices, :]
	fraud_train_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_train_target.csv', index=True)
	normal_train_indices = train_target[train_target.isFraud == 0].index
	normal_train_data = train_data.ix[normal_train_indices, :]
	normal_train_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_train_data.csv', index=True)
	normal_train_target = train_target.ix[normal_train_indices, :]
	normal_train_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_train_target.csv', index=True)

	test_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_data.csv', index=True)
	test_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_target.csv', index=True)
	fraud_test_indices = test_target[test_target.isFraud == 1].index.values
	fraud_test_data = test_data.ix[fraud_test_indices, :]
	fraud_test_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_test_data.csv', index=True)
	fraud_test_target = test_target.ix[fraud_test_indices, :]
	fraud_test_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_test_target.csv', index=True)
	normal_test_indices = test_target[test_target.isFraud == 0].index
	normal_test_data = test_data.ix[normal_test_indices, :]
	normal_test_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_test_data.csv', index=True)
	normal_test_target = test_target.ix[normal_test_indices, :]
	normal_test_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_test_target.csv', index=True)

print("Number transactions train dataset: ", format(len(train_data), ',d'))
print("Number transactions test dataset: ", format(len(test_data), ',d'))
print("Total number of transactions: ", format(len(train_data)+len(test_data), ',d'))
print("Number transactions train classifications: ", format(len(train_target), ',d'))
print("Number transactions test classifications: ", format(len(test_target), ',d'))
print("Total number of classifications: ", format(len(train_target)+len(test_target), ',d'))


# Perform Under sample of TRANSFER and CASH_OUT
print("\n## Perform Under sample of TRANSFER and CASH_OUT")
number_fraud_records = len(tmpData[tmpData.isFraud == 1])
# Picking the indices of the fraud and normal classes
fraud_indices = tmpData[tmpData.isFraud == 1].index.values
normal_indices = tmpData[tmpData.isFraud == 0].index
# Out of the indices we picked, randomly select "x" records, where "x" = total number of frauds
random_normal_indices = np.random.choice(normal_indices, number_fraud_records, replace=False)
random_normal_indices = np.array(random_normal_indices)
# Appending the 2 indices
under_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = tmpData.iloc[under_indices, :]
under_data = under_sample_data.ix[:, under_sample_data.columns != 'isFraud']												# not isFraud column, only data
under_target = under_sample_data.ix[:, under_sample_data.columns == 'isFraud']	 											# only isFraud column
# Showing ratio
print("\n## Fraud Ratio (TRANSFER and CASH_OUT) after **Data under sample**: ")
print("% of normal transactions: ", len(under_sample_data[under_sample_data.isFraud == 0])/len(under_sample_data))
print("% of fraud transactions: ", len(under_sample_data[under_sample_data.isFraud == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# Undersampled dataset
print("\n## Undersampled dataset:")
train_under_data, test_under_data, train_under_target, test_under_target = train_test_split(under_data, under_target, test_size=0.3, random_state=0)
if not os.path.isfile('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data.csv'):
	train_under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data.csv', index=True)
	train_under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_target.csv', index=True)
	fraud_train_under_indices = train_under_target[train_under_target.isFraud == 1].index.values
	fraud_train_under_data = train_under_data.ix[fraud_train_under_indices, :]
	fraud_train_under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_train_under_data.csv', index=True)
	fraud_train_under_target = train_under_target.ix[fraud_train_under_indices, :]
	fraud_train_under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_train_under_target.csv', index=True)
	normal_train_under_indices = train_under_target[train_under_target.isFraud == 0].index
	normal_train_under_data = train_under_data.ix[normal_train_indices, :]
	normal_train_under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_train_under_data.csv', index=True)
	normal_train_under_target = train_under_target.ix[normal_train_under_indices, :]
	normal_train_under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_train_under_target.csv', index=True)

	test_under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_under_data.csv', index=True)
	test_under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_under_target.csv', index=True)
	fraud_test_under_indices = test_under_target[test_under_target.isFraud == 1].index.values
	fraud_test_under_data = test_under_data.ix[fraud_test_under_indices, :]
	fraud_test_under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_test_under_data.csv', index=True)
	fraud_test_under_target = test_under_target.ix[fraud_test_under_indices, :]
	fraud_test_under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/fraud_test_under_target.csv', index=True)
	normal_test_under_indices = test_under_target[test_under_target.isFraud == 0].index
	normal_test_under_data = test_under_data.ix[normal_test_under_indices, :]
	normal_test_under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_test_under_data.csv', index=True)
	normal_test_under_target = test_under_target.ix[normal_test_under_indices, :]
	normal_test_under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/normal_test_under_target.csv', index=True)

print("Number transactions train dataset: ", format(len(train_under_data), ',d'))
print("Number transactions test dataset: ", format(len(test_under_data), ',d'))
print("Total number of transactions: ", format(len(train_under_data)+len(test_under_data), ',d'))
print("Number transactions train classifications: ", format(len(train_under_target), ',d'))
print("Number transactions test classifications: ", format(len(test_under_target), ',d'))
print("Total of classifications: ", format(len(train_under_target)+len(test_under_target), ',d'))


if not os.path.isfile('/media/thiago/ubuntu/datasets/fraudDetection/orig_boxcox.csv'):
	# ## saving whole data
	print("\n## saving whole data")
	# orig_boxcox
	print("# orig_boxcox")
	data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_boxcox.csv', index=True)
	# boxcox
	print("# boxcox")
	boxcox = data.copy()
	boxcox.drop(['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount'], axis=1, inplace=True)
	boxcox.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/boxcox.csv', index=True)
	# orig
	print("# orig")
	orig = data.copy()
	orig.drop(['amount_boxcox', 'oldbalanceOrg_boxcox', 'newbalanceOrg_boxcox', 'oldbalanceDest_boxcox', 'newbalanceDest_boxcox'], axis=1, inplace=True)
	orig.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig.csv', index=True)
	# target
	print("# target")
	target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/target.csv', index=True)
	# orig_PCA
	X = PCA().fit_transform(orig.values)
	df = pd.DataFrame(X, index=orig.index.values)
	df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_PCA.csv', index=True)
	# orig_PCA2
	X2 = PCA(n_components=2).fit_transform(orig.values)
	df2 = pd.DataFrame(X2, index=orig.index.values)
	df2.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_PCA2.csv', index=True)
else:
	data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_boxcox.csv', index_col=0)
	boxcox = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/boxcox.csv', index_col=0)
	orig = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig.csv', index_col=0)
	target = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/target.csv', index_col=0)
	orig_PCA = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_PCA.csv', index_col=0)
	orig_PCA2 = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_PCA2.csv', index_col=0)
# ## orig_tsne
# randomSeed = 13204
# X = TSNE(n_components=len(orig.columns), random_state=randomSeed).fit_transform(orig.values)
# df = pd.DataFrame(X, index=orig.index.values)
# df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_TSNE.csv', index=True)
# X2 = TSNE(n_components=2, random_state=randomSeed).fit_transform(orig.values)
# df2 = pd.DataFrame(X2, index=orig.values)
# df2.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_TSNE2.csv', index=True)


if not os.path.isfile('/media/thiago/ubuntu/datasets/fraudDetection/under_orig_boxcox.csv'):
	# ## saving under sample data
	print("\n## saving under sample data")
	# ## under_orig_boxcox
	print("# under_orig_boxcox")
	under_data.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_orig_boxcox.csv', index=True)
	# ## under_boxcox
	print("# under_boxcox")
	under_boxcox = under_data.copy()
	under_boxcox.drop(['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount'], axis=1, inplace=True)
	under_boxcox.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_boxcox.csv', index=True)
	# ## under_orig
	print("# under_orig")
	under_orig = under_data.copy()
	under_orig.drop(['amount_boxcox', 'oldbalanceOrg_boxcox', 'newbalanceOrg_boxcox', 'oldbalanceDest_boxcox', 'newbalanceDest_boxcox'], axis=1, inplace=True)
	under_orig.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_orig.csv', index=True)
	# ## under_target
	print("# under_target")
	under_target.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_target.csv', index=True)
	# ## under_orig_PCA
	X = PCA().fit_transform(under_orig.values)
	df = pd.DataFrame(X, index=under_orig.index.values)
	df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA.csv', index=True)
	# ## under_orig_PCA2
	X2 = PCA(n_components=2).fit_transform(under_orig.values)
	df2 = pd.DataFrame(X2, index=under_orig.index.values)
	df2.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA2.csv', index=True)
else:
	# ## loading under sample data
	print("\n## loading under sample data")
	under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_orig_boxcox.csv', index_col=0)
	under_boxcox = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_boxcox.csv', index_col=0)
	under_orig = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_orig.csv', index_col=0)
	under_target = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_target.csv', index_col=0)
	under_PCA = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA.csv', index_col=0)
	under_PCA2 = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA2.csv', index_col=0)
# ## under_tsne
# randomSeed = 13204
# X = TSNE(n_components=len(under_orig.columns), random_state=randomSeed).fit_transform(under_orig.values)
# df = pd.DataFrame(X, index=under_orig.index.values)
# df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_TSNE.csv', index=True)
# X2 = TSNE(n_components=2, random_state=randomSeed).fit_transform(under_orig.values)
# df2 = pd.DataFrame(X2, index=under_orig.values)
# df2.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_TSNE2.csv', index=True)


# # ToDo: Sparse Coding, with tunning of alpha (2 and 5), iterations (100 and 500), dictSize (100 and colmnNum)
# # ToDo: Denoising from dictionar learning
# miniBatch = MiniBatchDictionaryLearning(n_components=10, alpha=5, n_iter=100)
# dictionary = miniBatch.fit(data.values).components_
# sparseCode = miniBatch.transform(data.values)
# denoised = np.dot(sparseCode, dictionary)
# sparseCodedf = pd.DataFrame(sparseCode, index=data.index.values)
# denoiseddf = pd.DataFrame(denoised, index=data.index.values)
# sparseCodedf.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c10_it100.csv', index=True)
# denoiseddf.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c10_it100.csv', index=True)
# miniBatch = MiniBatchDictionaryLearning(n_components=10, alpha=5, n_iter=100)
# dictionary = miniBatch.fit(under_data.values).components_
# sparseCode = miniBatch.transform(under_data.values)
# denoised = np.dot(sparseCode, dictionary)
# sparseCodedf = pd.DataFrame(sparseCode, index=under_data.index.values)
# denoiseddf = pd.DataFrame(denoised, index=under_data.index.values)
# sparseCodedf.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c10_it100.csv', index=True)
# denoiseddf.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c10_it100.csv', index=True)


# ## 3. Logistic regression classifier #################################################################################
# From the model evaluation (or confusion matrix = https://en.wikipedia.org/wiki/Confusion_matrix), we know that:
#  1. Accuracy = (TP + TN) / Total
#  2. Presicion = TP / (TP + FP)
#  3. Recall = TP / (TP + FN)
# we are interested in the recall score to capture the most fraudulent transactions. 
# due to the imbalance of the data, many observations could be predicted as False Negatives. **Recall** captures this.
#######################################################################################################################

# Performing parameter estimation [LogisticRegression and GridSearchCV]
print("\n## Performing parameter estimation [LogisticRegression and GridSearchCV]")
start_time = time.time()
fold = KFold(n_splits=5, shuffle=True, random_state=777)
grid = {
	'C': np.array([100, 10, 1, 0.1, 0.01, 0.001]),
	'solver': ['newton-cg']
}
lr = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
gs = GridSearchCV(lr, grid, scoring='roc_auc', cv=fold)
gs.fit(train_under_data, train_under_target)
print ('# exec_time:', time.time() - start_time)
print ('# best_score_:', gs.best_score_)
print ('# best_params_:', gs.best_params_)


# Performing parameter estimation [LogisticRegressionCV]
print("\n## Performing parameter estimation [LogisticRegressionCV]")
start_time = time.time()
lrcv = LogisticRegressionCV(
	Cs=list(np.power(10.0, np.arange(-10, 10))),
	penalty='l2',
	scoring='roc_auc',
	cv=fold,
	random_state=777,
	max_iter=10000,
	fit_intercept=True,
	solver='newton-cg',
	tol=10
)
lrcv.fit(train_under_data, train_under_target)
print ('# exec_time:', time.time() - start_time)
print ('# max_auc_roc:', lrcv.scores_[1].mean(axis=0).max())
print ('# C_:', lrcv.C_)
best_c = lrcv.C_[0]
# best_c = 100

# Perfoming LogisticRegression [train=undersample and predict=undersample]
print("\n## Perfoming LogisticRegression [train=undersample and predict=undersample]")
lr = LogisticRegression(C=best_c, penalty='l1')
lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
predicted_unsample = lr.predict(test_under_data.values)
# confusion matrix
print("# Plot non-normalized confusion matrix [train=undersample and predict=undersample]")
cnf_matrix = confusion_matrix(test_under_target, predicted_unsample)
print("# Recall metric for undersample dataset: {0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
target_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
# plt.show()
# ROC CURVE
predicted_unsample_score = lr_fit.decision_function(test_under_data.values)
fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_unsample_score)
roc_auc = auc(fpr, tpr)
print("# ROC Curve:", roc_auc)
# plt.title('ROC Curve [train=undersample and predict=undersample]')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([-0.1, 1.0])
# plt.ylim([-0.1, 1.01])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


# Perfoming LogisticRegression [train=undersample and predict=complete]
print("\n## Perfoming LogisticRegression [train=undersample and predict=complete]")
predicted = lr.predict(test_data.values)
# Compute confusion matrix
print("# Plot non-normalized confusion matrix [train=undersample and predict=complete]")
cnf_matrix = confusion_matrix(test_target, predicted)
print("# Recall metric for whole dataset: {0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
target_names = [0, 1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
# plt.show()
# ROC CURVE
predicted_score = lr_fit.decision_function(test_data.values)
fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
roc_auc = auc(fpr, tpr)
print("# ROC Curve:", roc_auc)
# plt.title('ROC Curve [train=undersample and predict=complete]')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([-0.1, 1.0])
# plt.ylim([-0.1, 1.01])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


# ## LogisticRegression results
# print("\n## LogisticRegression results:")
# print("# estimator:")
# print(lr)
# print("# intercept:")
# print(lr.intercept_)
# print("# coefficient:")
# print(lr.coef_)
# print("# labels:")
# print(data.columns.tolist())


# ## Feature ranking
# print('\n## Feature Ranking')
# print_feature_ranking(train_under.values, train_under_target.values.ravel(), data.columns.tolist(), lr, "LogReg")


########################################################################################################################

# ToDo:
# rfe e identificar as features mais importantes
# testar com todos os dados
# testar com pca
# testar com t-sne
# testar com GridSearch
# What if we use all features (boxcox transformation and original data)
# Using SVC or other methodologies
# testar pca
# testar tsne
# testar dictionary learning
# usar os recursos de visualizacao do featureSelectio e de outro sobre fraud que plota duas distributions juntas
