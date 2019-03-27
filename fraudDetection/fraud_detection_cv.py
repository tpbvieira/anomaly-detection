# coding: utf-8
"""
This code implements:
	Feature extraction based on dictionaries and sparse coding via MiniBatchDictionaryLearning
	PR_AUC and ROC_AUC evaluation of original data and extracted features for fraud detection by logistic regression, SVM and LinearSVM
"""

from __future__ import division
import os.path
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import itertools
import time
from itertools import cycle
from sklearn import preprocessing
from scipy.stats import skew, boxcox
from statsmodels.tools import categorical
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLasso
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, auc, roc_auc_score, roc_curve, classification_report, average_precision_score
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.svm import SVC, LinearSVC
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
sns.set_style("dark")

fraud_data_path = '/media/thiago/ubuntu/datasets/fraud/'
results_file = open('results/fraud_detection_cv.txt', 'w')


## return string dateTime
def now_datetime_str():
	tmp_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	return tmp_time


## Read CSV file into a Panda DataFrame and print some information
def read_csv(csv):
	df = pd.read_csv(csv)
	print >> results_file, "{}: {} has {} observations and {} columns".format(now_datetime_str(), csv, df.shape[0], df.shape[1])
	print >> results_file, "{}: Column name checking::: {}".format(now_datetime_str(), df.columns.tolist())
	return df


## read dataframe and find the missing data on the columns and # of missing
def checking_missing(df):
	try:
		if isinstance(df, pd.DataFrame):
			df_na_bool = pd.concat([df.isnull().any(), df.isnull().sum(), (df.isnull().sum()/df.shape[0])*100], axis=1, keys=['df_bool', 'df_amt', 'missing_ratio_percent'])
			df_na_bool = df_na_bool.loc[df_na_bool['df_bool'] == True]
			return df_na_bool
		else:
			print >> results_file, "{}: The input is not panda DataFrame".format(now_datetime_str())

	except (UnboundLocalError, RuntimeError):
		print >> results_file, "{}: Something is wrong".format(now_datetime_str())


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


## Defines plot_confusion_matrix function, where true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
def plot_confusion_matrix(_cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""	This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.	"""
	plt.imshow(_cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	if normalize:
		_cm = _cm.astype('float') / _cm.sum(axis=1)[:, np.newaxis]
		print >> results_file, "Normalized confusion matrix"
	else:
		print >> results_file, 'Confusion matrix, without normalization'

	print >> results_file, _cm

	thresh = _cm.max() / 2.
	for i, j in itertools.product(range(_cm.shape[0]), range(_cm.shape[1])):
		plt.text(j, i, _cm[i, j], horizontalalignment="center", color="white" if _cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


## Read CSV file into a Panda DataFrame and print some information
print("## Loading data", file=results_file)

## complete
data = pd.read_csv(fraud_data_path + 'orig.csv', index_col=0) 							# selected features of raw data
# data = pd.read_csv(fraud_data_path + 'boxcox.csv', index_col=0)						# selected features of boxcox data
# data = pd.read_csv(fraud_data_path + 'orig_PCA.csv', index_col=0)						# PCA of selected features of raw data
# data = pd.read_csv(fraud_data_path + 'orig_PCA2.csv', index_col=0)						# 2 features-PCA of selected features of raw data
target = pd.read_csv(fraud_data_path + 'target.csv', index_col=0)

## under
under_data = pd.read_csv(fraud_data_path + 'under_orig.csv', index_col=0)				# selected features of raw data
# under_data = pd.read_csv(fraud_data_path + 'under_boxcox.csv', index_col=0)			# selected features of boxcox data
# under_data = pd.read_csv(fraud_data_path + 'under_PCA.csv', index_col=0)				# PCA of selected features of raw data
# under_data = pd.read_csv(fraud_data_path + 'under_PCA2.csv', index_col=0)				# 2 features-PCA of selected features of raw data
# under_data = pd.read_csv(fraud_data_path + 'under_TSNE.csv', index_col=0)
# under_data = pd.read_csv(fraud_data_path + 'under_TSNE2.csv', index_col=0)
under_target = pd.read_csv(fraud_data_path + 'under_target.csv', index_col=0)

## split data
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)
train_under_data, test_under_data, train_under_target, test_under_target = train_test_split(under_data, under_target, test_size=0.3, random_state=0)

## Perfoming SVM [complete/complete]
# SVC = LinearSVC()
# SVC = SVC()
# SVC_fit = SVC.fit(train_data, train_target)
# test_predicted = SVC.predict(test_data)
# ## ROC_AUC
# predicted_unsample_score = SVC_fit.decision_function(test_data.values)
# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_unsample_score)
# roc_auc = auc(fpr, tpr)
# ## Precision-Recall AUC
# precision = dict()
# recall = dict()
# average_precision = dict()
# n_classes = test_target.shape[1]
# for i in range(n_classes):
# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_unsample_score)
# 	average_precision[i] = average_precision_score(test_target, predicted_unsample_score)
# print("## [complete/complete]\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0]))

# ## Perfoming SVM [under/under]
# SVC = LinearSVC()
# # SVC = SVC()
# SVC_fit = SVC.fit(train_under_data, train_under_target)
# test_predicted = SVC.predict(test_under_data)
# ## ROC_AUC
# predicted_unsample_score = SVC_fit.decision_function(test_under_data.values)
# fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_unsample_score)
# roc_auc = auc(fpr, tpr)
# ## Precision-Recall AUC
# precision = dict()
# recall = dict()
# average_precision = dict()
# n_classes = test_target.shape[1]
# for i in range(n_classes):
# 	precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_unsample_score)
# 	average_precision[i] = average_precision_score(test_under_target, predicted_unsample_score)
# print >> results_file, "## [under/under]\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

# ## Perfoming SVM [under/complete]
# test_predicted = SVC.predict(test_data)
# ## ROC_AUC
# predicted_unsample_score = SVC_fit.decision_function(test_data.values)
# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_unsample_score)
# roc_auc = auc(fpr, tpr)
# ## Precision-Recall AUC
# precision = dict()
# recall = dict()
# average_precision = dict()
# n_classes = test_target.shape[1]
# for i in range(n_classes):
# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_unsample_score)
# 	average_precision[i] = average_precision_score(test_target, predicted_unsample_score)
# print >> results_file, "## [under/complete]\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

# ## Perfoming LogisticRegression [complete/complete]
# best_c = 100
# lr = LogisticRegression(C=best_c, penalty='l1')
# lr_fit = lr.fit(train_data, train_target.values.ravel())
# test_predicted = lr.predict(test_data.values)
# ## ROC_AUC
# predicted_score = lr_fit.decision_function(test_data.values)
# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
# roc_auc = auc(fpr, tpr)
# ## Precision-Recall AUC
# precision = dict()
# recall = dict()
# average_precision = dict()
# n_classes = test_target.shape[1]
# for i in range(n_classes):
# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
# 	average_precision[i] = average_precision_score(test_target, predicted_score)
# print >> results_file, "## [complete/complete]\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

# ## Perfoming LogisticRegression [under/under]
# lr = LogisticRegression(C=best_c, penalty='l1')
# lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
# test_under_predicted = lr.predict(test_under_data.values)
# ## ROC_AUC
# predicted_unsample_score = lr_fit.decision_function(test_under_data.values)
# fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_unsample_score)
# roc_auc = auc(fpr, tpr)
# ## Precision-Recall AUC
# precision = dict()
# recall = dict()
# average_precision = dict()
# n_classes = test_target.shape[1]
# for i in range(n_classes):
# 	precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_unsample_score)
# 	average_precision[i] = average_precision_score(test_under_target, predicted_unsample_score)
# print >> results_file, "## [under/under]\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

# ## Perfoming LogisticRegression [under/complete]
# test_predicted = lr.predict(test_data.values)
# ## ROC_AUC
# predicted_score = lr_fit.decision_function(test_data.values)
# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
# roc_auc = auc(fpr, tpr)
# ## Precision-Recall AUC
# precision = dict()
# recall = dict()
# average_precision = dict()
# n_classes = test_target.shape[1]
# for i in range(n_classes):
# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
# 	average_precision[i] = average_precision_score(test_target, predicted_score)
# print >> results_file, "## [under/complete]\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

## MiniBatch Dictionary Learning cross-validation
alpha_range = [1, 2, 3, 5, 7, 8, 10, 100] # sparsity
n_range = [2, 6, 10, 20, 40, 100] # dictionary size
for a in alpha_range:
	for n in n_range:

		if a > n:
			continue;

		it = 100;
		best_c = 100 # previously calculated through cross validation code for logisctic regression

		## train_data
		if not os.path.isfile(fraud_data_path + 'train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating train_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating train_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			train_data_std = preprocessing.scale(train_data.values)
			train_data = pd.DataFrame(train_data_std, index=train_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(train_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'train_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(train_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=train_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=train_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'train_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
		
		## test_data
		if not os.path.isfile(fraud_data_path + 'test_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating test_data_denoised_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating test_data_denoised_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			test_data_std = preprocessing.scale(test_data.values)
			test_data = pd.DataFrame(test_data_std, index=test_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(test_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'test_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(test_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=test_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'test_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=test_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'test_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		## normal_train_data
		if not os.path.isfile(fraud_data_path + 'normal_train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating normal_train_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating normal_train_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			normal_train_data = pd.read_csv(fraud_data_path + 'normal_train_data.csv', index_col=0)
			normal_train_data_std = preprocessing.scale(normal_train_data.values)
			normal_train_data = pd.DataFrame(normal_train_data_std, index=normal_train_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(normal_train_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'normal_train_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(normal_train_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=normal_train_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'normal_train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=normal_train_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'normal_train_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		## fraud_train_data
		if not os.path.isfile(fraud_data_path + 'fraud_train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating fraud_train_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating fraud_train_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			fraud_train_data = pd.read_csv(fraud_data_path + 'fraud_train_data.csv', index_col=0)
			fraud_train_data_std = preprocessing.scale(fraud_train_data.values)
			fraud_train_data = pd.DataFrame(fraud_train_data_std, index=fraud_train_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(fraud_train_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'fraud_train_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(fraud_train_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=fraud_train_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'fraud_train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=fraud_train_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'fraud_train_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		## train_under_data
		if not os.path.isfile(fraud_data_path + 'train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating train_under_data_denoised_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating train_under_data_denoised_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			train_under_data_std = preprocessing.scale(train_under_data.values)
			train_under_data = pd.DataFrame(train_under_data_std, index=train_under_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(train_under_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'train_under_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(train_under_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=train_under_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=train_under_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'train_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		## test_under_data
		if not os.path.isfile(fraud_data_path + 'test_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating test_under_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating test_under_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			test_under_data_std = preprocessing.scale(test_under_data.values)
			test_under_data = pd.DataFrame(test_under_data_std, index=test_under_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(test_under_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'test_under_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(test_under_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=test_under_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'test_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=test_under_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'test_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		## normal_train_under_data
		if not os.path.isfile(fraud_data_path + 'normal_train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating normal_train_under_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating normal_train_under_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			normal_train_under_data = pd.read_csv(fraud_data_path + 'normal_train_under_data.csv', index_col=0)
			normal_train_under_data_std = preprocessing.scale(normal_train_under_data.values)
			normal_train_under_data = pd.DataFrame(normal_train_under_data_std, index=normal_train_under_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(normal_train_under_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'normal_train_under_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(normal_train_under_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=normal_train_under_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'normal_train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=normal_train_under_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'normal_train_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		## fraud_train_under_data
		if not os.path.isfile(fraud_data_path + 'fraud_train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			print('## Creating fraud_train_under_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
			print >> results_file, '## Creating fraud_train_under_data_sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
			fraud_train_under_data = pd.read_csv(fraud_data_path + 'fraud_train_under_data.csv', index_col=0)
			normal_train_under_data_std = preprocessing.scale(fraud_train_under_data.values)
			fraud_train_under_data = pd.DataFrame(normal_train_under_data_std, index=fraud_train_under_data.index.values)
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(fraud_train_under_data.values).components_
			dictionary_df = pd.DataFrame(dictionary)
			dictionary_df.to_csv(fraud_data_path + 'fraud_train_under_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			sparseCode = miniBatch.transform(fraud_train_under_data.values)
			sparseCode_df = pd.DataFrame(sparseCode, index=fraud_train_under_data.index.values)
			sparseCode_df.to_csv(fraud_data_path + 'fraud_train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised = np.dot(sparseCode, dictionary)
			denoised_df = pd.DataFrame(denoised, index=fraud_train_under_data.index.values)
			denoised_df.to_csv(fraud_data_path + 'fraud_train_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

		# ## dictionary data
		# str_data_type = '## denoised_a{:d}_c{:d}_it{:d}'.format(a,n,it)
		# train_data = pd.read_csv(fraud_data_path + 'train_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		# test_data = pd.read_csv(fraud_data_path + 'test_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		# train_under_data = pd.read_csv(fraud_data_path + 'train_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		# test_under_data = pd.read_csv(fraud_data_path + 'test_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)

		# ## Perfoming LogisticRegression [complete/complete]		
		# lr = LogisticRegression(C=best_c, penalty='l1')
		# lr_fit = lr.fit(train_data, train_target.values.ravel())
		# test_predicted = lr.predict(test_data.values)
		# ## ROC AUC
		# predicted_score = lr_fit.decision_function(test_data.values)
		# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		# roc_auc = auc(fpr, tpr)
		# ## Precision-Recall AUC
		# precision = dict()
		# recall = dict()
		# average_precision = dict()
		# n_classes = test_target.shape[1]
		# for i in range(n_classes):
		# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		# 	average_precision[i] = average_precision_score(test_target, predicted_score)
		# str_comp_comp =  "\t[complete/complete]" + "\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

		# ## Perfoming LogisticRegression [under/under]
		# lr = LogisticRegression(C=best_c, penalty='l1')
		# lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
		# test_under_predicted = lr.predict(test_under_data.values)
		# ## ROC AUC
		# predicted_under_score = lr_fit.decision_function(test_under_data.values)
		# fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_under_score)
		# roc_auc = auc(fpr, tpr)
		# # Precision-Recall AUC
		# precision = dict()
		# recall = dict()
		# average_precision = dict()
		# n_classes = test_under_target.shape[1]
		# for i in range(n_classes):
		# 	precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_under_score)
		# 	average_precision[i] = average_precision_score(test_under_target, predicted_under_score)
		# str_under_under = "\t[under/under]" + "\tROC_AUC:{0:.4f}".format(roc_auc) + " PR_AUC:{:.4f}".format(average_precision[0])

		# ## Perfoming LogisticRegression [under/complete]
		# test_predicted = lr.predict(test_data.values)
		# ## ROC AUC
		# predicted_score = lr_fit.decision_function(test_data.values)
		# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		# roc_auc = auc(fpr, tpr)
		# ## Precision-Recall AUC
		# precision = dict()
		# recall = dict()
		# average_precision = dict()
		# for i in range(n_classes):
		# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		# 	average_precision[i] = average_precision_score(test_target, predicted_score)
		# str_under_comp = "\t[under/complete]" + "\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])
		# print >> results_file, str_data_type + str_comp_comp + str_under_under + str_under_comp

		# ## sparse data
		# str_data_type = '## sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it)
		# train_data = pd.read_csv(fraud_data_path + 'train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		# test_data = pd.read_csv(fraud_data_path + 'test_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		# train_under_data = pd.read_csv(fraud_data_path + 'train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		# test_under_data = pd.read_csv(fraud_data_path + 'test_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)

		# ## Perfoming LogisticRegression [complete/complete]
		# lr = LogisticRegression(C=best_c, penalty='l1')
		# lr_fit = lr.fit(train_data, train_target.values.ravel())
		# test_predicted = lr.predict(test_data.values)		
		# ## ROC AUC
		# predicted_score = lr_fit.decision_function(test_data.values)
		# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		# roc_auc = auc(fpr, tpr)
		# ## Precision-Recall AUC
		# precision = dict()
		# recall = dict()
		# average_precision = dict()
		# n_classes = test_target.shape[1]
		# for i in range(n_classes):
		# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		# 	average_precision[i] = average_precision_score(test_target, predicted_score)
		# str_comp_comp =  "\t[complete/complete]" + "\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

		# ## Perfoming LogisticRegression [under/under]
		# lr = LogisticRegression(C=best_c, penalty='l1')
		# lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
		# test_under_predicted = lr.predict(test_under_data.values)
		# ## ROC AUC
		# predicted_under_score = lr_fit.decision_function(test_under_data.values)
		# fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_under_score)
		# roc_auc = auc(fpr, tpr)
		# ## Precision-Recall AUC
		# precision = dict()
		# recall = dict()
		# average_precision = dict()
		# n_classes = test_under_target.shape[1]
		# for i in range(n_classes):
		# 	precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_under_score)
		# 	average_precision[i] = average_precision_score(test_under_target, predicted_under_score)
		# str_under_comp = "\t[under/under]" + "\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])

		# ## Perfoming LogisticRegression [under/complete]
		# test_predicted = lr.predict(test_data.values)
		# ## ROC AUC
		# predicted_score = lr_fit.decision_function(test_data.values)
		# fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		# roc_auc = auc(fpr, tpr)
		# ## Precision-Recall AUC
		# precision = dict()
		# recall = dict()
		# average_precision = dict()
		# for i in range(n_classes):
		# 	precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		# 	average_precision[i] = average_precision_score(test_target, predicted_score)
		# str_under_comp =  "\t[under/complete]" + "\tROC_AUC:{0:.4f}".format(roc_auc) + "\tPR_AUC:{:.4f}".format(average_precision[0])
		# print >> results_file, str_data_type + str_comp_comp + str_under_under + str_under_comp

results_file.close()