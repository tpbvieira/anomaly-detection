# coding: utf-8
####################################################################################################################################################################
## Cross-Validation for evaluating the reconstruction performance by dictionary learning methods and parameter configurations applied to a fraud detection dataset
####################################################################################################################################################################

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matlab.engine
import warnings
from time import time
from sklearn import preprocessing
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.measure import (compare_mse, compare_nrmse)
from sklearn.linear_model import orthogonal_mp
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, auc, roc_auc_score, roc_curve, classification_report, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
## Prints a comparison of matrices
def print_vector_comparison(X, Y, comparison_name=None):
	difference = X - Y
	norm = np.sqrt(np.sum(difference ** 2))
	mse = compare_mse(X, Y)
	nrmse = compare_nrmse(X, Y)
	cos = cosine_similarity(X, Y)
	if comparison_name is not None:
		printing_text = comparison_name + ':Norm:%(NORM).4f\tMSE:%(MSE).4f\tNRMSE:%(NRMSE).4f\tCos:%(COS).4f' % {'NORM': norm, 'MSE': mse, 'NRMSE': nrmse, 'COS': cos}
		print(printing_text)
		# print >> results, printing_text
	return norm, mse, nrmse, cos

## settings
file_path = '/media/thiago/ubuntu/datasets/fraudDetection/'
results = open('results/fraud_detection_reconstruction_dl_cv.txt', 'w')
# matlab = matlab.engine.start_matlab()

## Load data
print('## Loading saved data...')
print >> results, '## Loading saved data...'
# data = pd.read_csv(file_path + 'train_under_data.csv', index_col=0)
# test_data = pd.read_csv(file_path + 'test_under_data.csv', index_col=0)
test_data = pd.read_csv(file_path + 'test_data.csv', index_col=0)
# test_target = pd.read_csv(file_path + 'test_under_target.csv', index_col=0)
test_target = pd.read_csv(file_path + 'test_target.csv', index_col=0)
# print >> results, 'Data Shape: ' + str(data.shape)
# matlab.cd(r'/home/thiago/dev/projects/anomaly-detection/distributed-tensor-dictionary-learning', nargout=0)
# matlab.project(nargout=0)

## parameters
L = test_data.shape[0]
N = test_data.shape[1]
tnz_range = [1, 2, 3, 5]
K_range = [2, 6, 10, 20, 40, 100]
noIt_range = [100]

## discard hearder and index
test_data = test_data.values
test_target = test_target.values

## Load previously calculated data and evaluates their reconstruction performance for each parameter configuration
for K in K_range:
	for tnz in tnz_range:
		if tnz > K:
			continue
		for noIt in noIt_range:
			transform_algorithms = [
				('MiniBatchDL_OMP', 'omp', {'transform_n_nonzero_coefs': tnz})
				# , ('MOD_javaORMP', '', {})
				# , ('RLS-DLA_javaORMP', '', {})
				# , ('K-SVD_javaORMP', '', {})
				# , ('T-MOD_javaORMP', '', {})
				# , ('K-HOSVD', '', {})
				]
			for title, transform_algorithm, kwargs in transform_algorithms:
				predicted_score_norm = np.zeros([L, 1])
				predicted_score_mse = np.zeros([L, 1])
				predicted_score_nrmse = np.zeros([L, 1])
				predicted_score_cos = np.zeros([L, 1])
				fraud_dictionary = pd.read_csv(file_path + 'fraud_train_under_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(tnz, K, noIt), index_col=0).values
				normal_dictionary = pd.read_csv(file_path + 'normal_train_under_data_dictionary_a{:d}_c{:d}_it{:d}.csv'.format(tnz, K, noIt), index_col=0).values
				for l in range(0, L):					
					str_subject = title + ':L:{:d}_N:{:d}_tnz:{:d}_K:{:d}_noIt:{:d}_l:{:d}_fraud:{:d}'.format(L, N, tnz, K, noIt, l, test_target[l,0])
					
					fraud_sparseCode = orthogonal_mp(np.transpose(fraud_dictionary), test_data[l,:], precompute=True, n_nonzero_coefs=tnz)
					fraud_recov = np.dot(fraud_sparseCode, fraud_dictionary)
					fraud_norm, fraud_mse, fraud_nrmse, fraud_cos = print_vector_comparison(test_data[l,:], fraud_recov)

					normal_sparseCode = orthogonal_mp(np.transpose(normal_dictionary), test_data[l,:], precompute=True, n_nonzero_coefs=tnz)
					normal_recov = np.dot(normal_sparseCode, normal_dictionary)
					normal_norm, normal_mse, normal_nrmse, normal_cos = print_vector_comparison(test_data[l,:], normal_recov)

					if fraud_norm < normal_norm:
						predicted_score_norm[l,0] = 1
					
					if fraud_mse < normal_mse:
						predicted_score_mse[l,0] = 1
					
					if fraud_nrmse < normal_nrmse:
						predicted_score_nrmse[l,0] = 1
					
					if fraud_cos > normal_cos:
						predicted_score_cos[l,0] = 1

				## Norm
				## ROC AUC
				fpr, tpr, thresholds = roc_curve(test_target, predicted_score_norm)
				roc_auc = auc(fpr, tpr)
				## Precision-Recall AUC
				precision = dict()
				recall = dict()
				average_precision = dict()
				n_classes = test_target.shape[1]
				for i in range(n_classes):
					precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score_norm)
					average_precision[i] = average_precision_score(test_target, predicted_score_norm)
				str_comp =  title + ':L:{:d}_N:{:d}_tnz:{:d}_K:{:d}_noIt:{:d}'.format(L, N, tnz, K, noIt) + "\tNORM_ROC_AUC:{0:.4f}".format(roc_auc) + "\tNORM_PR_AUC:{:.4f}".format(average_precision[0])
				print(str_comp)
				print >> results, str_comp

				## MSE
				## ROC AUC
				fpr, tpr, thresholds = roc_curve(test_target, predicted_score_mse)
				roc_auc = auc(fpr, tpr)
				## Precision-Recall AUC
				precision = dict()
				recall = dict()
				average_precision = dict()
				n_classes = test_target.shape[1]
				for i in range(n_classes):
					precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score_mse)
					average_precision[i] = average_precision_score(test_target, predicted_score_mse)
				str_comp =  title + ':L:{:d}_N:{:d}_tnz:{:d}_K:{:d}_noIt:{:d}'.format(L, N, tnz, K, noIt) + "\tMSE_ROC_AUC:{0:.4f}".format(roc_auc) + "\tMSE_PR_AUC:{:.4f}".format(average_precision[0])
				print(str_comp)
				print >> results, str_comp

				## NRMSE
				## ROC AUC
				fpr, tpr, thresholds = roc_curve(test_target, predicted_score_nrmse)
				roc_auc = auc(fpr, tpr)
				## Precision-Recall AUC
				precision = dict()
				recall = dict()
				average_precision = dict()
				n_classes = test_target.shape[1]
				for i in range(n_classes):
					precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score_nrmse)
					average_precision[i] = average_precision_score(test_target, predicted_score_nrmse)
				str_comp =  title + ':L:{:d}_N:{:d}_tnz:{:d}_K:{:d}_noIt:{:d}'.format(L, N, tnz, K, noIt) + "\tNRMSE_ROC_AUC:{0:.4f}".format(roc_auc) + "\tNRMSE_PR_AUC:{:.4f}".format(average_precision[0])
				print(str_comp)
				print >> results, str_comp

				## COS
				## ROC AUC
				fpr, tpr, thresholds = roc_curve(test_target, predicted_score_cos)
				roc_auc = auc(fpr, tpr)
				## Precision-Recall AUC
				precision = dict()
				recall = dict()
				average_precision = dict()
				n_classes = test_target.shape[1]
				for i in range(n_classes):
					precision[i], recall[i], _ = precision_recall_curve(test_target, predicted_score_cos)
					average_precision[i] = average_precision_score(test_target, predicted_score_cos)
				str_comp =  title + ':L:{:d}_N:{:d}_tnz:{:d}_K:{:d}_noIt:{:d}'.format(L, N, tnz, K, noIt) + "\tCOS_ROC_AUC:{0:.4f}".format(roc_auc) + "\tCOS_PR_AUC:{:.4f}".format(average_precision[0])
				print(str_comp)
				print >> results, str_comp
results.close()