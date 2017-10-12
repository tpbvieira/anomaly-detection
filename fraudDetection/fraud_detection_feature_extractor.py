# coding: utf-8
########################################################################################################################
## EDA, Parameter Estimation and Feature extraction for Fraud detection
########################################################################################################################

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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, auc, roc_auc_score, roc_curve, classification_report, average_precision_score
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
sns.set_style("dark")

dataset_path = '/media/thiago/ubuntu/datasets/fraudDetection/'


## return string dateTime
def now_datetime_str():
	tmp_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	return tmp_time


## Read CSV file into a Panda DataFrame and print some information
def read_csv(csv):
	df = pd.read_csv(csv)
	# print("{}: {} has {} observations and {} columns".format(now_datetime_str(), csv, df.shape[0], df.shape[1]))
	# print("{}: Column name checking::: {}".format(now_datetime_str(), df.columns.tolist()))
	return df


## function to read dataframe and find the missing data on the columns and # of missing
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


## Plots a Correlation Heatmap
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


## Read CSV file into a Panda DataFrame and print some information
print("## Loading data")
raw_data = read_csv(dataset_path + 'Synthetic_Financial_Datasets_For_Fraud_Detection.csv')


## ### 1. EDA (exploratory data analysis ) ##############################################################################
# In this section, we will do EDA to understand the data more. From the simulation, there are 5 transaction types as per
#  illustrated below.


## Check if there's any null values.
# print("## Let's check the dataset if there's any null values.")
# print(checking_missing(raw_data))


## Look at the dataset sample and other properties.
# print("## Head:")
# print(raw_data.head(5))
# print("## Describe:")
# print(raw_data.describe())
# print("## Info:")
# print(raw_data.info())


## Plot transaction count by transaction type
# print("## Plot transaction count by transaction type:")
# f, ax = plt.subplots(1, 1, figsize=(8, 8))
# raw_data.type.value_counts().plot(kind='bar', title="Transaction count by transaction type", ax=ax, figsize=(8,8))
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()


# # *isFraud* is the indicator which indicates the actual fraud transactions
# # *isFlaggedFraud* is what the system prevents the transaction due to some thresholds being triggered.
# # Plot Fraud (1) and Legitmate (0) transactions count by transaction type
# print("## Plot Fraud (1) and Legitmate (0) transactions count by transaction type:")
# ax = raw_data.groupby(['type', 'isFraud']).size().plot(kind='bar')
# ax.set_title("Fraud (1) and Legitmate (0) transactions count by transaction type")
# ax.set_xlabel("(Type, isFraud)")
# ax.set_ylabel("Count of transaction")
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()
# # Observation: Only TRANSFER and CASH_OUT transactions have fraud


## Plot transactions which are flagged as isFlaggedFraud per transaction type
# print("## Plot Flagged Fraud (1) and Legitmate (0) transactions count by transaction type:")
# ax = raw_data.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')
# ax.set_title("Flagged Fraud (1) and Legitmate (0) transactions count by transaction type")
# ax.set_xlabel("(Type, isFlaggedFraud)")
# ax.set_ylabel("Count of transaction")
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()
# # Observation: the system can flag only 16 transfer transactions as fraud.


## Plot fraud *TRANSFER* analysis
# print("## Plot fraud *TRANSFER* analysis")
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
# print("## Plot fraud *CASH_OUT* analysis")
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


## 2. Modeling ######################################################################################################
# focus only on **TRANSFER** and **CASH_OUT** (where there are fraud), data slicing and data transformation. Keep only interested transaction type ('TRANSFER', 'CASH_OUT')
print("## Focus only on **TRANSFER** and **CASH_OUT** (where there are fraud)")
tmpData = raw_data.loc[(raw_data['type'].isin(['TRANSFER', 'CASH_OUT'])), :]


# Data slicing - Drop unnecessary data ('step', 'nameOrig', 'nameDest', 'isFlaggedFraud')
print("## Data slicing - Drop unnecessary data ('step', 'nameOrig', 'nameDest', 'isFlaggedFraud')")
# tmpData.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
tmpData.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
tmpData = tmpData.reset_index(drop=True)
# Convert categorical variables to numeric variable
a = np.array(tmpData['type'])
b = categorical(a, drop=True)
tmpData['type_num'] = b.argmax(1)
tmpData.drop(['type'], axis=1, inplace=True)


## Plot Correlations of TRANSFER and CASH_OUT transactions and selected features
# print("## Plot Correlations of TRANSFER and CASH_OUT transactions and selected features")
# plotCorrelationHeatmap(tmpData, "TRANSFER and CASH_OUT Correlation")
# plotCorrelationHeatmap(raw_data.loc[(raw_data.type == 'TRANSFER'), :], "TRANSFER Correlation")
# plotCorrelationHeatmap(raw_data.loc[(raw_data.type == 'CASH_OUT'), :], "CASH_OUT Correlation")


## Quickly get the count and the target variable count.
# print("## Plot Transaction count by type")
# ax = tmpData.type.value_counts().plot(kind='bar', title="Transaction count by type", figsize=(6,6))
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# plt.show()
# print("## Plot Fraud (1) and Legitmate (0) transactions count")
# ax = pd.value_counts(tmpData['isFraud'], sort = True).sort_index().plot(kind='bar', title="Fraud (1) and Legitmate (0)
#  transactions count")
# for p in ax.patches:
# 	ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))
# plt.show()


## 2.1 Feature extraction ############################################################################################
# Based on the dataset, the numeric variables are quite skew, in this case. I will try to scale it with 2 methods
# (SQRT and Box-Cox) and compare them on the graph.
# skewness of the distribution: Normally distributed data has skewness should be about 0. A skewness value > 0 means
# that there is more weight in the left tail of the distribution
# Boxcox transformation: makes the data normal
print("## Extracting Box-Cox features ")
# print("## Plot Transformations for **amount**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.title("Amount Histogram")
# plt.hist(tmpData['amount'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("Transaction amount")
# plt.text(10,100000,"Skewness: {0:.2f}".format(skew(tmpData['amount'])))
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


# print("## Plot Transformations for **oldbalanceOrg**:")
# figure = plt.figure(figsize=(16, 5))
# figure.add_subplot(131)
# plt.title("oldbalanceOrg Histogram")
# plt.hist(tmpData['oldbalanceOrg'] ,facecolor='blue',alpha=0.75)
# plt.xlabel("old balance originated")
# plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmpData['oldbalanceOrg'])))
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


# print("## Plot Transformations for **newbalanceOrg**:")
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


# print("## Plot Transformations for **oldbalanceDest**:")
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


# print("## Plot Transformations for **newbalanceDest**:")
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
# print("## Feature Selection: Use only the box-cox data transformation for prediction")
# tmpData.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','amount'],axis=1,inplace=True)
# print(tmpData.head(5))
# Plot Correlations of TRANSFER and CASH_OUT transactions and box-cox features
# print("## Plot Correlations of TRANSFER and CASH_OUT transactions and box-cox features")
# sns.heatmap(tmpData)
# plt.show()
# sns.heatmap(tmpData.loc[(tmpData.type_num == 0), :].corr())
# plt.show()
# sns.heatmap(tmpData.loc[(tmpData.type_num == 1), :].corr())
# plt.show()

# print("## Fraud Ratio
#  (TRANSFER and CASH_OUT):")
# print("% of normal transactions: ", len(tmpData[tmpData.isFraud == 0])/len(tmpData))
# print("% of fraud transactions: ", len(tmpData[tmpData.isFraud == 1])/len(tmpData))
# print("Total number of transactions in resampled data: ", len(tmpData))
# # There're only actual fraud of 0.3% and this is very imbalance data.


# I will under sample the dataset by creating a 50-50 ratio of randomly selecting 'x' amount of sample from majority
# class, with 'x' being the total number of records with the minority class.
# Number of data points in the minority class

## Preparing data for training

## Whole dataset
print("## Splitting dataset for training...")
data = tmpData.ix[:, tmpData.columns != 'isFraud']
target = tmpData.ix[:, tmpData.columns == 'isFraud']
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)


## Saves train and test of complete data if they dont exist
if not os.path.isfile(dataset_path + 'train_data.csv'):
	train_data.to_csv(dataset_path + 'train_data.csv', index=True)
	train_target.to_csv(dataset_path + 'train_target.csv', index=True)
if not os.path.isfile(dataset_path + 'fraud_train_data.csv'):
	fraud_train_indices = train_target[train_target.isFraud == 1].index.values
	fraud_train_data = train_data.ix[fraud_train_indices, :]
	fraud_train_data.to_csv(dataset_path + 'fraud_train_data.csv', index=True)
	fraud_train_target = train_target.ix[fraud_train_indices, :]
	fraud_train_target.to_csv(dataset_path + 'fraud_train_target.csv', index=True)
if not os.path.isfile(dataset_path + 'normal_train_data.csv'):
	normal_train_indices = train_target[train_target.isFraud == 0].index.values
	normal_train_data = train_data.ix[normal_train_indices, :]
	normal_train_data.to_csv(dataset_path + 'normal_train_data.csv', index=True)
	normal_train_target = train_target.ix[normal_train_indices, :]
	normal_train_target.to_csv(dataset_path + 'normal_train_target.csv', index=True)
if not os.path.isfile(dataset_path + 'test_data.csv'):
	test_data.to_csv(dataset_path + 'test_data.csv', index=True)
	test_target.to_csv(dataset_path + 'test_target.csv', index=True)
if not os.path.isfile(dataset_path + 'fraud_test_data.csv'):
	fraud_test_indices = test_target[test_target.isFraud == 1].index.values
	fraud_test_data = test_data.ix[fraud_test_indices, :]
	fraud_test_data.to_csv(dataset_path + 'fraud_test_data.csv', index=True)
	fraud_test_target = test_target.ix[fraud_test_indices, :]
	fraud_test_target.to_csv(dataset_path + 'fraud_test_target.csv', index=True)
if not os.path.isfile(dataset_path + 'normal_test_data.csv'):
	normal_test_indices = test_target[test_target.isFraud == 0].index.values
	normal_test_data = test_data.ix[normal_test_indices, :]
	normal_test_data.to_csv(dataset_path + 'normal_test_data.csv', index=True)
	normal_test_target = test_target.ix[normal_test_indices, :]
	normal_test_target.to_csv(dataset_path + 'normal_test_target.csv', index=True)


## Saves extracted feaures if it does not exist or load the saved data
print("## Saves extracted feaures if it does not exist or load the saved data")
if not os.path.isfile(dataset_path + 'orig_boxcox.csv'):
	## orig_boxcox
	data.to_csv(dataset_path + 'orig_boxcox.csv', index=True)
	
	## boxcox (selected features)
	boxcox = data.copy()
	boxcox.drop(['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount'], axis=1, inplace=True)
	boxcox.to_csv(dataset_path + 'boxcox.csv', index=True)
	
	## orig (selected features)
	orig = data.copy()
	orig.drop(['amount_boxcox', 'oldbalanceOrg_boxcox', 'newbalanceOrg_boxcox', 'oldbalanceDest_boxcox', 'newbalanceDest_boxcox'], axis=1, inplace=True)
	orig.to_csv(dataset_path + 'orig.csv', index=True)
	
	## target
	target.to_csv(dataset_path + 'target.csv', index=True)
	
	## orig_PCA
	X = PCA().fit_transform(orig.values)
	df = pd.DataFrame(X, index=orig.index.values)
	df.to_csv(dataset_path + 'orig_PCA.csv', index=True)
	
	## orig_PCA2 (PCA with 2 components)
	X2 = PCA(n_components=2).fit_transform(orig.values)
	df2 = pd.DataFrame(X2, index=orig.index.values)
	df2.to_csv(dataset_path + 'orig_PCA2.csv', index=True)
else:
	data = pd.read_csv(dataset_path + 'orig_boxcox.csv', index_col=0)
	boxcox = pd.read_csv(dataset_path + 'boxcox.csv', index_col=0)
	orig = pd.read_csv(dataset_path + 'orig.csv', index_col=0)
	target = pd.read_csv(dataset_path + 'target.csv', index_col=0)
	orig_PCA = pd.read_csv(dataset_path + 'orig_PCA.csv', index_col=0)
	orig_PCA2 = pd.read_csv(dataset_path + 'orig_PCA2.csv', index_col=0)


## Perform Under sample of TRANSFER and CASH_OUT
print("## Perform Undersample")
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
under_data = under_sample_data.ix[:, under_sample_data.columns != 'isFraud'] # not isFraud column, only data
under_target = under_sample_data.ix[:, under_sample_data.columns == 'isFraud'] # only isFraud column


## Extract Undersampled dataset for train and test data. Saves into files if thei dont exists
print("## Splitting undersampled dataset for training...")
train_under_data, test_under_data, train_under_target, test_under_target = train_test_split(under_data, under_target, test_size=0.3, random_state=0)
if not os.path.isfile(dataset_path + 'train_under_data.csv'):
	train_under_data.to_csv(dataset_path + 'train_under_data.csv', index=True)
	train_under_target.to_csv(dataset_path + 'train_under_target.csv', index=True)
if not os.path.isfile(dataset_path + 'fraud_train_under_data.csv'):
	fraud_train_under_indices = train_under_target[train_under_target.isFraud == 1].index.values
	fraud_train_under_data = train_under_data.ix[fraud_train_under_indices, :]
	fraud_train_under_data.to_csv(dataset_path + 'fraud_train_under_data.csv', index=True)
	fraud_train_under_target = train_under_target.ix[fraud_train_under_indices, :]
	fraud_train_under_target.to_csv(dataset_path + 'fraud_train_under_target.csv', index=True)
if not os.path.isfile(dataset_path + 'normal_train_under_data.csv'):
	normal_train_under_indices = train_under_target[train_under_target.isFraud == 0].index
	normal_train_under_data = train_under_data.ix[normal_train_under_indices, :]
	normal_train_under_data.to_csv(dataset_path + 'normal_train_under_data.csv', index=True)
	normal_train_under_target = train_under_target.ix[normal_train_under_indices, :]
	normal_train_under_target.to_csv(dataset_path + 'normal_train_under_target.csv', index=True)
if not os.path.isfile(dataset_path + 'test_under_data.csv'):
	test_under_data.to_csv(dataset_path + 'test_under_data.csv', index=True)
	test_under_target.to_csv(dataset_path + 'test_under_target.csv', index=True)
if not os.path.isfile(dataset_path + 'fraud_test_under_data.csv'):
	fraud_test_under_indices = test_under_target[test_under_target.isFraud == 1].index.values
	fraud_test_under_data = test_under_data.ix[fraud_test_under_indices, :]
	fraud_test_under_data.to_csv(dataset_path + 'fraud_test_under_data.csv', index=True)
	fraud_test_under_target = test_under_target.ix[fraud_test_under_indices, :]
	fraud_test_under_target.to_csv(dataset_path + 'fraud_test_under_target.csv', index=True)
if not os.path.isfile(dataset_path + 'normal_test_under_data.csv'):
	normal_test_under_indices = test_under_target[test_under_target.isFraud == 0].index
	normal_test_under_data = test_under_data.ix[normal_test_under_indices, :]
	normal_test_under_data.to_csv(dataset_path + 'normal_test_under_data.csv', index=True)
	normal_test_under_target = test_under_target.ix[normal_test_under_indices, :]
	normal_test_under_target.to_csv(dataset_path + 'normal_test_under_target.csv', index=True)


## Saves extracted feaures if it does not exist or load the saved data
print("## Saves extracted feaures if it does not exist or load the saved data")
if not os.path.isfile(dataset_path + 'under_orig_boxcox.csv'):
	## under_orig_boxcox
	under_data.to_csv(dataset_path + 'under_orig_boxcox.csv', index=True)
	
	## under_boxcox
	under_boxcox = under_data.copy()
	under_boxcox.drop(['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount'], axis=1, inplace=True)
	under_boxcox.to_csv(dataset_path + 'under_boxcox.csv', index=True)
	
	## under_orig
	under_orig = under_data.copy()
	under_orig.drop(['amount_boxcox', 'oldbalanceOrg_boxcox', 'newbalanceOrg_boxcox', 'oldbalanceDest_boxcox', 'newbalanceDest_boxcox'], axis=1, inplace=True)
	under_orig.to_csv(dataset_path + 'under_orig.csv', index=True)
	
	## under_target[]
	under_target.to_csv(dataset_path + 'under_target.csv', index=True)
	
	## under_orig_PCA
	X = PCA().fit_transform(under_orig.values)
	df = pd.DataFrame(X, index=under_orig.index.values)
	df.to_csv(dataset_path + 'under_PCA.csv', index=True)
	
	## under_orig_PCA2
	X2 = PCA(n_components=2).fit_transform(under_orig.values)
	df2 = pd.DataFrame(X2, index=under_orig.index.values)
	df2.to_csv(dataset_path + 'under_PCA2.csv', index=True)
else:
	under_data = pd.read_csv(dataset_path + 'under_orig_boxcox.csv', index_col=0)
	under_boxcox = pd.read_csv(dataset_path + 'under_boxcox.csv', index_col=0)
	under_orig = pd.read_csv(dataset_path + 'under_orig.csv', index_col=0)
	under_target = pd.read_csv(dataset_path + 'under_target.csv', index_col=0)
	under_PCA = pd.read_csv(dataset_path + 'under_PCA.csv', index_col=0)
	under_PCA2 = pd.read_csv(dataset_path + 'under_PCA2.csv', index_col=0)