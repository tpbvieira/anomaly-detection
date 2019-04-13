# coding=utf-8
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, recall_score, precision_score, precision_recall_curve
from algorithms import fit, \
    cv_location_contamination, cv_location_threshold, cv_skewness_contamination, cv_kurtosis_contamination, cv_skewness_threshold,\
    predict_by_location_contamination, predict_by_location_threshold, predict_by_location_centered_contamination, \
    predict_by_skewness_contamination, predict_by_skewness_centered_contamination, predict_by_skewness_threshold, predict_by_skewness_centered_threshold, \
    predict_by_kurtosis_contamination, predict_by_kurtosis_centered_contamination, predict_by_kurtosis_threshold, predict_by_kurtosis_centered_threshold
from data_processing import get_dataset, split_dataset
from plots.plots import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")


start_time = time.time()
prefixes = ['0.15', '0.25', '1', '2']
for prefix in prefixes:
    print('')
    print('Dataset Version: %s' % prefix)
    print('')
    dataset_list, files_list = get_dataset(prefix)
    pre_processed_dataset = split_dataset(dataset_list)

    num_datasets = len(dataset_list)
    for i in range(num_datasets):
        training_df = pre_processed_dataset['training'][i]
        cv_df = pre_processed_dataset['cross_validation'][i]
        testing_df = pre_processed_dataset['testing'][i]

        actual_labels = np.array(testing_df['Label'])
        file_name = files_list[i].name.replace('/', '').replace('.', '')

        # Training
        L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(np.array(training_df, dtype=float))

        best_contamination = cv_location_contamination(cv_df, rob_mean, rob_precision)
        pred_label, mahal_dist = predict_by_location_contamination(testing_df, rob_mean, rob_precision, best_contamination)
        print('%s - predict_by_location_contamination - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))

        best_threshold = cv_location_threshold(cv_df, rob_mean, rob_precision, rob_dist)
        pred_label, mahal_dist = predict_by_location_threshold(testing_df, rob_mean, rob_precision, best_threshold)
        print('%s - predict_by_location_threshold - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))

        best_contamination = cv_skewness_contamination(cv_df, rob_skew, rob_precision)
        pred_label = predict_by_skewness_contamination(testing_df, rob_precision, rob_skew, best_contamination)
        print('%s - predict_by_skewness_contamination - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))
        pred_label  = predict_by_skewness_centered_contamination(testing_df, rob_precision, rob_skew, best_contamination)
        print('%s - predict_by_skewness_centered_contamination - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))

        best_threshold = cv_skewness_threshold(cv_df, rob_skew, rob_precision, rob_skew_dist)
        pred_label = predict_by_skewness_threshold(testing_df, rob_precision, rob_skew, best_threshold)
        print('%s - predict_by_skewness_threshold - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))

        best_contamination = cv_kurtosis_contamination(cv_df, rob_kurt, rob_precision)
        pred_label = predict_by_kurtosis_contamination(testing_df, rob_precision, rob_kurt, best_contamination)
        print('%s - predict_by_kurtosis_contamination - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))
        pred_label  = predict_by_kurtosis_centered_contamination(testing_df, rob_precision, rob_kurt, best_contamination)
        print('%s - predict_by_kurtosis_centered_contamination - F1: %f' % (files_list[i].name, f1_score(actual_labels, pred_label)))

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
    print("--- Tempo de execução %s segundos ---" % (time.time() - start_time))
