# coding: utf-8
########################################################################################################################
# ## EDA, Parameter Estimatio and Feature Ranking for Fraud detection
# 
# The provided data has the financial transaction data as well as the target variable **isFraud**, which is the actual
# fraud status of the transaction and **isFlaggedFraud** is the indicator which the simulation is used to flag the
# transaction using some threshold. The goal should be how we can improve and come up with better threshold to capture
# the fraud transaction.
########################################################################################################################

from __future__ import division
import os.path
import warnings
import datetime
import numpy as np  																									# linear algebra
import pandas as pd  																									# data processing, CSV file I/O (e.g. pd.read_csv)
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
from mpl_toolkits.mplot3d import Axes3D
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


# In[24]:
# Defines plot_confusion_matrix function
# true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
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
data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/boxcox.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_PCA.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/orig_PCA2.csv', index_col=0)
target = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/target.csv', index_col=0)

under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_orig.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_boxcox.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA2.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_TSNE.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_TSNE2.csv', index_col=0)
under_target = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_target.csv', index_col=0)


train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)
train_under_data, test_under_data, train_under_target, test_under_target = train_test_split(under_data, under_target, test_size=0.3, random_state=0)

alpha_range = [1, 2, 3, 5, 7, 8, 10, 100]
n_range = [2, 6, 10, 20, 40, 100, 200]

for a in alpha_range:
	for n in n_range:
		it = 100;

		if not os.path.isfile('/media/thiago/ubuntu/datasets/fraudDetection/train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it)):
			# # ToDo: Sparse Coding, with tunning of alpha (2 and 5), iterations (100 and 500), dictSize (100 and colmnNum)
			# # ToDo: Denoising from dictionar learning
			print('## create a{:d}_c{:d}_it{:d}'.format(a,n,it))
			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(train_data.values).components_
			sparseCode = miniBatch.transform(train_data.values)
			denoised = np.dot(sparseCode, dictionary)
			sparseCode_df = pd.DataFrame(sparseCode, index=train_data.index.values)
			sparseCode_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised_df = pd.DataFrame(denoised, index=train_data.index.values)
			denoised_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(test_data.values).components_
			sparseCode = miniBatch.transform(test_data.values)
			denoised = np.dot(sparseCode, dictionary)
			sparseCode_df = pd.DataFrame(sparseCode, index=test_data.index.values)
			sparseCode_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised_df = pd.DataFrame(denoised, index=test_data.index.values)
			denoised_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(train_under_data.values).components_
			sparseCode = miniBatch.transform(train_under_data.values)
			denoised = np.dot(sparseCode, dictionary)
			sparseCode_df = pd.DataFrame(sparseCode, index=train_under_data.index.values)
			sparseCode_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised_df = pd.DataFrame(denoised, index=train_under_data.index.values)
			denoised_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)

			miniBatch = MiniBatchDictionaryLearning(n_components=n, alpha=a, n_iter=100)
			dictionary = miniBatch.fit(test_under_data.values).components_
			sparseCode = miniBatch.transform(test_under_data.values)
			denoised = np.dot(sparseCode, dictionary)
			sparseCode_df = pd.DataFrame(sparseCode, index=test_under_data.index.values)
			sparseCode_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)
			denoised_df = pd.DataFrame(denoised, index=test_under_data.index.values)
			denoised_df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index=True)


		print('## denoised_a{:d}_c{:d}_it{:d}'.format(a,n,it))
		train_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		test_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		train_under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		test_under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_under_data_denoised_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)


		best_c = 100


		# Perfoming LogisticRegression [complete/complete]
		print("\n## [complete/complete]")
		lr = LogisticRegression(C=best_c, penalty='l1')
		lr_fit = lr.fit(train_data, train_target.values.ravel())
		test_predicted = lr.predict(test_data.values)
		# confusion matrix
		#print("# Plot non-normalized confusion matrix [complete/complete]")
		cnf_matrix = confusion_matrix(test_target, test_predicted)
		print("# Recall:\t{0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
		# target_names = [0,1]
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
		# plt.show()
		# # ROC CURVE
		predicted_score = lr_fit.decision_function(test_data.values)
		fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		roc_auc = auc(fpr, tpr)
		print("# ROC_AUC:\t{0:.4f}".format(roc_auc))
		# plt.title('ROC Curve [complete/complete]')
		# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'r--')
		# plt.xlim([-0.1, 1.0])
		# plt.ylim([-0.1, 1.01])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

		# Compute Precision-Recall and plot curve
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		lw = 2
		precision = dict()
		recall = dict()
		average_precision = dict()
		n_classes = test_target.shape[1]
		for i in range(n_classes):
		    precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		    average_precision[i] = average_precision_score(test_target, predicted_score)
		print("# AP:\t{:.4f}".format(average_precision[0]))
		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall[0], precision[0], lw=lw, color='navy', label='PRC, AP={:.4f} and AUC={:.4f} of complete/complete'.format(average_precision[0], roc_auc))


		# Perfoming LogisticRegression [under/under]
		print("\n## [under/under]")
		lr = LogisticRegression(C=best_c, penalty='l1')
		lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
		test_under_predicted = lr.predict(test_under_data.values)
		# confusion matrix
		#print("# Plot non-normalized confusion matrix [under/under]")
		cnf_matrix = confusion_matrix(test_under_target, test_under_predicted)
		print("# Recall:\t{0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
		# target_names = [0,1]
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
		# plt.show()
		# # ROC CURVE
		predicted_under_score = lr_fit.decision_function(test_under_data.values)
		fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_under_score)
		roc_auc = auc(fpr, tpr)
		print("# ROC_AUC:\t{0:.4f}".format(roc_auc))
		# plt.title('ROC Curve [under/under]')
		# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'r--')
		# plt.xlim([-0.1, 1.0])
		# plt.ylim([-0.1, 1.01])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

		# Compute Precision-Recall and plot curve
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		lw = 2
		precision = dict()
		recall = dict()
		average_precision = dict()
		n_classes = test_under_target.shape[1]
		for i in range(n_classes):
		    precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_under_score)
		    average_precision[i] = average_precision_score(test_under_target, predicted_under_score)
		print("# AP:\t{:.4f}".format(average_precision[0]))
		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall[0], precision[0], lw=lw, color='navy', label='PRC, AP={:.4f} and AUC={:.4f} of Under/Under'.format(average_precision[0], roc_auc))


		# Perfoming LogisticRegression [under/complete]
		print("\n## [under/complete]")
		test_predicted = lr.predict(test_data.values)
		# Compute confusion matrix
		# print("# Plot non-normalized confusion matrix [under/complete]")
		cnf_matrix = confusion_matrix(test_target, test_predicted)
		print("# Recall:\t{0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
		# target_names = [0, 1]
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
		# plt.show()
		# ROC CURVE
		predicted_score = lr_fit.decision_function(test_data.values)
		fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		roc_auc = auc(fpr, tpr)
		print("# ROC_AUC:\t{0:.4f}".format(roc_auc))
		# plt.title('ROC Curve [under/complete]')
		# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'r--')
		# plt.xlim([-0.1, 1.0])
		# plt.ylim([-0.1, 1.01])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

		# Compute Precision-Recall and plot curve
		precision = dict()
		recall = dict()
		average_precision = dict()
		for i in range(n_classes):
		    precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		    average_precision[i] = average_precision_score(test_target, predicted_score)
		print("# AP:\t{:.4f}".format(average_precision[0]))
		# Plot Precision-Recall curve
		plt.plot(recall[0], precision[0], lw=lw, color='turquoise', label='PRC, AP={:.4f} and AUC={:.4f} of Under/Whole'.format(average_precision[0], roc_auc))
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision-Recall Curve (PRC)')
		plt.legend(loc="lower left")
		plt.savefig('results/denoised_a{:d}_c{:d}_it{:d}.eps', bbox_inches='tight')
		# plt.show()


		print('## sparse_a{:d}_c{:d}_it{:d}'.format(a,n,it))
		train_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		test_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		train_under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)
		test_under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/test_under_data_sparse_a{:d}_c{:d}_it{:d}.csv'.format(a,n,it), index_col=0)


		best_c = 100


		# Perfoming LogisticRegression [complete/complete]
		print("\n## [complete/complete]")
		lr = LogisticRegression(C=best_c, penalty='l1')
		lr_fit = lr.fit(train_data, train_target.values.ravel())
		test_predicted = lr.predict(test_data.values)
		# confusion matrix
		#print("# Plot non-normalized confusion matrix [complete/complete]")
		cnf_matrix = confusion_matrix(test_target, test_predicted)
		print("# Recall:\t{0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
		# target_names = [0,1]
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
		# plt.show()
		# # ROC CURVE
		predicted_score = lr_fit.decision_function(test_data.values)
		fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		roc_auc = auc(fpr, tpr)
		print("# ROC_AUC:\t{0:.4f}".format(roc_auc))
		# plt.title('ROC Curve [complete/complete]')
		# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'r--')
		# plt.xlim([-0.1, 1.0])
		# plt.ylim([-0.1, 1.01])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

		# Compute Precision-Recall and plot curve
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		lw = 2
		precision = dict()
		recall = dict()
		average_precision = dict()
		n_classes = test_target.shape[1]
		for i in range(n_classes):
		    precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		    average_precision[i] = average_precision_score(test_target, predicted_score)
		print("# AP:\t{:.4f}".format(average_precision[0]))
		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall[0], precision[0], lw=lw, color='navy', label='PRC, AP={:.4f} and AUC={:.4f} of complete/complete'.format(average_precision[0], roc_auc))


		# Perfoming LogisticRegression [under/under]
		print("\n## [under/under]")
		lr = LogisticRegression(C=best_c, penalty='l1')
		lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
		test_under_predicted = lr.predict(test_under_data.values)
		# confusion matrix
		#print("# Plot non-normalized confusion matrix [under/under]")
		cnf_matrix = confusion_matrix(test_under_target, test_under_predicted)
		print("# Recall:\t{0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
		# target_names = [0,1]
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
		# plt.show()
		# # ROC CURVE
		predicted_under_score = lr_fit.decision_function(test_under_data.values)
		fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_under_score)
		roc_auc = auc(fpr, tpr)
		print("# ROC_AUC:\t{0:.4f}".format(roc_auc))
		# plt.title('ROC Curve [under/under]')
		# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'r--')
		# plt.xlim([-0.1, 1.0])
		# plt.ylim([-0.1, 1.01])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

		# Compute Precision-Recall and plot curve
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		lw = 2
		precision = dict()
		recall = dict()
		average_precision = dict()
		n_classes = test_under_target.shape[1]
		for i in range(n_classes):
		    precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_under_score)
		    average_precision[i] = average_precision_score(test_under_target, predicted_under_score)
		print("# AP:\t{:.4f}".format(average_precision[0]))
		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall[0], precision[0], lw=lw, color='navy', label='PRC, AP={:.4f} and AUC={:.4f} of Under/Under'.format(average_precision[0], roc_auc))


		# Perfoming LogisticRegression [under/complete]
		print("\n## [under/complete]")
		test_predicted = lr.predict(test_data.values)
		# Compute confusion matrix
		# print("# Plot non-normalized confusion matrix [under/complete]")
		cnf_matrix = confusion_matrix(test_target, test_predicted)
		print("# Recall:\t{0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
		# target_names = [0, 1]
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
		# plt.show()
		# ROC CURVE
		predicted_score = lr_fit.decision_function(test_data.values)
		fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
		roc_auc = auc(fpr, tpr)
		print("# ROC_AUC:\t{0:.4f}".format(roc_auc))
		# plt.title('ROC Curve [under/complete]')
		# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'r--')
		# plt.xlim([-0.1, 1.0])
		# plt.ylim([-0.1, 1.01])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

		# Compute Precision-Recall and plot curve
		precision = dict()
		recall = dict()
		average_precision = dict()
		for i in range(n_classes):
		    precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score)
		    average_precision[i] = average_precision_score(test_target, predicted_score)
		print("# AP:\t{:.4f}".format(average_precision[0]))
		# Plot Precision-Recall curve
		plt.plot(recall[0], precision[0], lw=lw, color='turquoise', label='PRC, AP={:.4f} and AUC={:.4f} of Under/Whole'.format(average_precision[0], roc_auc))
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision-Recall Curve (PRC)')
		plt.legend(loc="lower left")
		plt.savefig('results/sparse_a{:d}_c{:d}_it{:d}.eps', bbox_inches='tight')
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
# testar com pca
# testar com t-sne
# testar com GridSearch
# What if we use all features (boxcox transformation and original data)
# Using SVC or other methodologies
# testar pca
# testar tsne
# testar dictionary learning
# usar os recursos de visualizacao do featureSelectio e de outro sobre fraud que plota duas distributions juntas
# testar MOS com entropy
# testar análise com tipos separados
