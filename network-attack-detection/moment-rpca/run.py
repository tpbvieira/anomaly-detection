# coding=utf-8
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, recall_score, precision_score, precision_recall_curve
from algorithms import fit, \
    cv_location_contamination, cv_location_threshold, cv_skewness_contamination, cv_kurtosis_contamination, \
    cv_skewness_threshold, cv_kurtosis_threshold, \
    predict_by_location_contamination, predict_by_location_centered_contamination, predict_by_location_threshold, \
    predict_by_skewness_contamination, predict_by_skewness_centered_contamination, predict_by_skewness_threshold, predict_by_skewness_centered_threshold, \
    predict_by_kurtosis_contamination, predict_by_kurtosis_centered_contamination, predict_by_kurtosis_threshold, predict_by_kurtosis_centered_threshold
from data_processing import get_dataset, split_dataset
from plots.plots import plot_confusion_matrix
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

result_file_path = 'moment_rpca.txt'
result_file = open(result_file_path, 'w') # 'w' = clear all and write

start_time = time.time()
prefixes = ['0.15','0.25']
for prefix in prefixes:
    print('### Dataset Version: %s' % prefix, file=result_file)
    result_file.flush()
    dataset_list, files_list = get_dataset(prefix)
    pre_processed_dataset = split_dataset(dataset_list)

    num_datasets = len(dataset_list)
    for dataset in range(num_datasets):
        raw_train_df = pre_processed_dataset['training'][dataset]
        raw_cv_df = pre_processed_dataset['cross_validation'][dataset]
        raw_test_df = pre_processed_dataset['testing'][dataset]

        actual_labels = np.array(raw_test_df['Label'])
        file_name = files_list[dataset].name.replace('/', '').replace('.', '')

        cv_labels = raw_cv_df['Label']
        test_labels = raw_test_df['Label']

        raw_train_df = raw_train_df.drop(['Label'], axis=1)
        raw_cv_df = raw_cv_df.drop(['Label'], axis=1)
        raw_test_df = raw_test_df.drop(['Label'], axis=1)

        training_df = raw_train_df
        cv_df = raw_cv_df
        testing_df = raw_test_df

        col_list = list(raw_train_df.columns.values)
        best_f1_scenario = 0
        best_cols_scenario = None
        # while len(col_list) > 1:
        #     best_f1 = 0
        #     worse_f1 = 1
        #     best_cols = None
        #     worse_cols = None
        #     for col in col_list:
        #         n_col_list = col_list.copy()
        #         n_col_list.remove(col)
        #
        #         print(n_col_list, file=result_file)
        #         training_df = raw_train_df[n_col_list]
        #         cv_df = raw_cv_df[n_col_list]
        #         testing_df = raw_test_df[n_col_list]
        #
        #         # Training
        #         L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(np.array(training_df, dtype=float))
        #
        #         # Location CV and prediction
        #         best_contamination = cv_location_contamination(cv_df, cv_labels, rob_mean, rob_precision)
        #         pred_label = predict_by_location_contamination(testing_df, rob_mean, rob_precision, best_contamination)
        #         l_f1 = f1_score(actual_labels, pred_label)
        #         print('%s - predict_by_location_contamination - F1: %f' % (files_list[dataset].name, l_f1), file=result_file)
        #         # pred_label = predict_by_location_centered_contamination(testing_df, rob_mean, rob_precision, best_contamination)
        #         # print('%s - predict_by_location_centered_contamination - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         # best_threshold = cv_location_threshold(cv_df, cv_labels, rob_mean, rob_precision, rob_dist)
        #         # pred_label = predict_by_location_threshold(testing_df, rob_mean, rob_precision, best_threshold)
        #         # print('%s - predict_by_location_threshold - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         result_file.flush()
        #
        #         # Skewness CV and prediction
        #         best_contamination = cv_skewness_contamination(cv_df, cv_labels, rob_skew, rob_precision)
        #         pred_label = predict_by_skewness_contamination(testing_df, rob_precision, rob_skew, best_contamination)
        #         s_f1 = f1_score(actual_labels, pred_label)
        #         print('%s - predict_by_skewness_contamination - F1: %f' % (files_list[dataset].name, s_f1), file=result_file)
        #         # pred_label = predict_by_skewness_centered_contamination(testing_df, rob_precision, rob_skew, best_contamination)
        #         # print('%s - predict_by_skewness_centered_contamination - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         # best_threshold = cv_skewness_threshold(cv_df, cv_labels, rob_skew, rob_precision, rob_skew_dist)
        #         # pred_label = predict_by_skewness_threshold(testing_df, rob_precision, rob_skew, best_threshold)
        #         # print('%s - predict_by_skewness_threshold - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         # pred_label = predict_by_skewness_centered_threshold(testing_df, rob_precision, rob_skew, best_threshold)
        #         # print('%s - predict_by_skewness_centered_threshold - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         result_file.flush()
        #
        #         # Kurtosis CV and prediction
        #         best_contamination = cv_kurtosis_contamination(cv_df, cv_labels, rob_kurt, rob_precision)
        #         pred_label = predict_by_kurtosis_contamination(testing_df, rob_precision, rob_kurt, best_contamination)
        #         k_f1 = f1_score(actual_labels, pred_label)
        #         print('%s - predict_by_kurtosis_contamination - F1: %f' % (files_list[dataset].name, k_f1), file=result_file)
        #         # pred_label = predict_by_kurtosis_centered_contamination(testing_df, rob_precision, rob_kurt, best_contamination)
        #         # print('%s - predict_by_kurtosis_centered_contamination - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         # best_threshold = cv_kurtosis_threshold(cv_df, cv_labels, rob_kurt, rob_precision, rob_kurt_dist)
        #         # pred_label = predict_by_kurtosis_threshold(testing_df, rob_precision, rob_kurt, best_threshold)
        #         # print('%s - predict_by_kurtosis_threshold - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         # pred_label = predict_by_kurtosis_centered_threshold(testing_df, rob_precision, rob_kurt, best_threshold)
        #         # print('%s - predict_by_kurtosis_centered_threshold - F1: %f' % (files_list[dataset].name, f1_score(actual_labels, pred_label)), file=result_file)
        #         result_file.flush()
        #
        #         if l_f1 > best_f1:
        #             best_f1 = l_f1
        #             best_cols = n_col_list.copy()
        #         if s_f1 > best_f1:
        #             best_f1 = s_f1
        #             best_cols = n_col_list.copy()
        #         if k_f1 > best_f1:
        #             best_f1 = k_f1
        #             best_cols = n_col_list.copy()
        #
        #     col_list = best_cols
        #     if best_f1 >= best_f1_scenario:
        #         best_f1_scenario = best_f1
        #         best_cols_scenario = best_cols.copy()
        #         print('###',best_f1_scenario, best_cols_scenario, file=result_file)

        # test ROBPCA-AO from saved files
        robpca_result_file = 'robpca_k10_%s_test_df' % file_name
        if os.path.isfile(robpca_result_file):
            # print(robpca_result_file)
            robpca_test_pred = pd.read_csv(robpca_result_file, header=None)
            robpca_test_pred = robpca_test_pred[0]

            robpca_test_pred[robpca_test_pred == True] = -1
            robpca_test_pred[robpca_test_pred == False] = 0
            robpca_test_pred[robpca_test_pred == -1] = 1
            print('%s - robpca_k10_%s_test_df - F1: %f' % (
                files_list[dataset].name, file_name, f1_score(test_labels, robpca_test_pred)))

                # # Save Confusion Matrix
                # conf_matrix = plot_confusion_matrix(actual_labels, pred_label,
                #                                     np.array(['Normal Traffic', 'Anomaly']),
                #                                     normalize=True, title='Normalized confusion matrix')
                # np.set_printoptions(precision=2)
                # plt.savefig('plots/confusion_matrices/%s/%s' % (prefix, file_name))
                # plt.close()
                # # Save precision-recall curve
                # precision, recall, threshold = precision_recall_curve(actual_labels, pred_label)
                # fig, ax = plt.subplots()
                # ax.set(title='Precision-Recall Curve', ylabel='Recall', xlabel='Precision')
                # plt.plot(precision, recall, marker='.')
                # plt.savefig('plots/precision_recall_curve/%s/%s' % (prefix, file_name))
                # plt.close()
    print("--- Tempo de execução %s segundos ---" % (time.time() - start_time), file=result_file)
result_file.close()