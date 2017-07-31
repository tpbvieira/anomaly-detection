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

# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a2_c100_it100.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c100_it100.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c2_it500.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c10_it10.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c10_it100.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c100_it10.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/sparse_a5_c100_it500.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a2_c100_it100.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c100_it100.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c100_it10.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c2_it500.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c10_it10.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c10_it100.csv', index_col=0)
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/denoised_a5_c100_it500.csv', index_col=0)
target = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/target.csv', index_col=0)

under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_orig.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_boxcox.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_PCA2.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_TSNE.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_TSNE2.csv', index_col=0)

# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a2_c100_it100.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c100_it100.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c2_it500.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c10_it10.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c10_it100.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c100_it10.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underSparse_a5_c100_it500.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a2_c100_it100.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c100_it100.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c100_it10.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c2_it500.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c10_it10.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c10_it100.csv', index_col=0)
# under_data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/underDenoised_a5_c100_it500.csv', index_col=0)
under_target = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/under_target.csv', index_col=0)

# print("\n# Recovered Whole Data:")
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)
# print("Number transactions train dataset: ", format(len(train_data), ',d'))
# print("Number transactions test dataset: ", format(len(test_data), ',d'))
# print("Total number of transactions: ", format(len(train_data)+len(test_data), ',d'))
# print("Number transactions train classifications: ", format(len(train_target), ',d'))
# print("Number transactions test classifications: ", format(len(test_target), ',d'))
# print("Total number of classifications: ", format(len(train_target)+len(test_target), ',d'))

# print("\n# Recovered Under Sample Data:")
train_under_data, test_under_data, train_under_target, test_under_target = train_test_split(under_data, under_target, test_size=0.3, random_state=0)
# print("Number transactions train dataset: ", format(len(train_under), ',d'))
# print("Number transactions test dataset: ", format(len(test_under), ',d'))
# print("Total number of transactions: ", format(len(train_under)+len(test_under), ',d'))
# print("Number transactions train classifications: ", format(len(train_under_target), ',d'))
# print("Number transactions test classifications: ", format(len(test_under_target), ',d'))
# print("Total of classifications: ", format(len(train_under_target)+len(test_under_target), ',d'))


# # ToDo: t-SNE
# randomSeed = 13204
# X = TSNE(n_components=len(data.columns), random_state=randomSeed).fit_transform(data.values)
# df = pd.DataFrame(X, index=data.index.values)
# df.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/origTSNE.csv', index=True)
# X2 = TSNE(n_components=2, random_state=randomSeed).fit_transform(data.values)
# df2 = pd.DataFrame(X2, index=data.index.values)
# df2.to_csv('/media/thiago/ubuntu/datasets/fraudDetection/origTSNE2.csv', index=True)


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
# 
# From the model evaluation (or confusion matrix = https://en.wikipedia.org/wiki/Confusion_matrix), we know that:
#  1. Accuracy = (TP + TN) / Total
#  2. Presicion = TP / (TP + FP)
#  3. Recall = TP / (TP + FN)
# 
# we are interested in the recall score to capture the most fraudulent transactions. 
# due to the imbalance of the data, many observations could be predicted as False Negatives. **Recall** captures this.


# # Performing parameter estimation [LogisticRegression and GridSearchCV]
# print("\n## Performing parameter estimation [LogisticRegression and GridSearchCV]")
# start_time = time.time()
# fold = KFold(n_splits=5, shuffle=True, random_state=777)
# grid = {
# 	'C': np.array([100, 10, 1, 0.1, 0.01, 0.001]),
# 	'solver': ['newton-cg']
# }
# lr = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
# gs = GridSearchCV(lr, grid, scoring='roc_auc', cv=fold)
# gs.fit(train_under, train_under_target)
# print ('# exec_time:', time.time() - start_time)
# print ('# best_score_:', gs.best_score_)
# print ('# best_params_:', gs.best_params_)
#
#
# # Performing parameter estimation [LogisticRegressionCV]
# print("\n## Performing parameter estimation [LogisticRegressionCV]")
# start_time = time.time()
# lrcv = LogisticRegressionCV(
# 	Cs=list(np.power(10.0, np.arange(-10, 10))),
# 	penalty='l2',
# 	scoring='roc_auc',
# 	cv=fold,
# 	random_state=777,
# 	max_iter=10000,
# 	fit_intercept=True,
# 	solver='newton-cg',
# 	tol=10
# )
# lrcv.fit(train_under, train_under_target)
# print ('# exec_time:', time.time() - start_time)
# print ('# max_auc_roc:', lrcv.scores_[1].mean(axis=0).max())
# print ('# C_:', lrcv.C_)
# best_c = lrcv.C_[0]
best_c = 100


# Perfoming LogisticRegression [train=undersample and predict=undersample]
print("\n## [train=undersample, predict=undersample]")
lr = LogisticRegression(C=best_c, penalty='l1')
lr_fit = lr.fit(train_under_data, train_under_target.values.ravel())
test_under_predicted = lr.predict(test_under_data.values)
# confusion matrix
#print("# Plot non-normalized confusion matrix [train=undersample and predict=undersample]")
cnf_matrix = confusion_matrix(test_under_target, test_under_predicted)
print("# Recall: {0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
# target_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
# plt.show()
# # ROC CURVE
predicted_unsample_score = lr_fit.decision_function(test_under_data.values)
fpr, tpr, thresholds = roc_curve(test_under_target.values.ravel(), predicted_unsample_score)
roc_auc = auc(fpr, tpr)
print("# ROC_AUC: {0:.4f}".format(roc_auc))
# plt.title('ROC Curve [train=undersample and predict=undersample]')
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
    precision[i], recall[i], _ = precision_recall_curve(test_under_target, predicted_unsample_score)
    average_precision[i] = average_precision_score(test_under_target, predicted_unsample_score)
# Plot Precision-Recall curve
plt.clf()
plt.plot(recall[0], precision[0], lw=lw, color='navy', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.4f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.show()


# Perfoming LogisticRegression [train=undersample and predict=complete]
print("\n## [train=undersample and predict=complete]")
test_predicted = lr.predict(test_data.values)
# Compute confusion matrix
# print("# Plot non-normalized confusion matrix [train=undersample and predict=complete]")
cnf_matrix = confusion_matrix(test_target, test_predicted)
print("# Recall: {0:.4f}".format(cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))
# target_names = [0, 1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion Matrix')
# plt.show()
# ROC CURVE
predicted_score = lr_fit.decision_function(test_data.values)
fpr, tpr, thresholds = roc_curve(test_target.values.ravel(), predicted_score)
roc_auc = auc(fpr, tpr)
print("# ROC_AUC: {0:.4f}".format(roc_auc))
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
