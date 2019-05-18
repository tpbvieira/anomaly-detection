# coding=utf-8
import os
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from data_processing import get_dataset, split_dataset
from algorithms import fit, \
    cv_location_contamination, cv_location_threshold, cv_skewness_contamination, cv_kurtosis_contamination, cv_skewness_threshold, cv_kurtosis_threshold, \
    predict_by_location_contamination, predict_by_location_centered_contamination, predict_by_location_threshold, \
    predict_by_skewness_contamination, predict_by_skewness_centered_contamination, predict_by_skewness_threshold, predict_by_skewness_centered_threshold, \
    predict_by_kurtosis_contamination, predict_by_kurtosis_centered_contamination, predict_by_kurtosis_threshold, predict_by_kurtosis_centered_threshold
warnings.filterwarnings("ignore")

result_file_path = 'moment_rpca.txt'
result_file = open(result_file_path, 'w') # 'w' = clear all and write
prefixes = ['0.25', '0.15', '0.1']

start_time = time.time()
for prefix in prefixes:
    print('### Dataset Version: %s' % prefix, file=result_file)
    result_file.flush()

    dataset_list, files_list = get_dataset(prefix)
    pre_processed_dataset = split_dataset(dataset_list)
    num_datasets = len(dataset_list)
    col_list = ['n_dports>1024', 'flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address', 'n_sports<1024',
                'n_s_na_p_address', 'n_udp', 'n_d_na_p_address', 'n_d_a_p_address', 'n_s_c_p_address',
                'n_d_c_p_address', 'n_dports<1024', 'n_d_b_p_address', 'n_tcp']

    # col_list = ['flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address', 'n_sports<1024', 'n_s_na_p_address',
    #             'n_s_c_p_address', 'n_d_c_p_address', 'normal_flow_count', 'n_dports<1024']

    for dataset in range(num_datasets):
        file_name = files_list[dataset].name.replace('/', '').replace('.', '')
        raw_train_df = pre_processed_dataset['training'][dataset]
        raw_cv_df = pre_processed_dataset['cross_validation'][dataset]
        raw_test_df = pre_processed_dataset['testing'][dataset]

        # save labels for cv and testing
        actual_labels = np.array(raw_test_df['Label'])
        cv_labels = raw_cv_df['Label']
        test_labels = raw_test_df['Label']

        # drop labels
        raw_train_df = raw_train_df.drop(['Label'], axis=1)
        raw_cv_df = raw_cv_df.drop(['Label'], axis=1)
        raw_test_df = raw_test_df.drop(['Label'], axis=1)

        # select columns
        # col_list = raw_train_df.columns
        training_df = raw_train_df[col_list]
        cv_df = raw_cv_df[col_list]
        testing_df = raw_test_df[col_list]

        # Training
        L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(np.array(training_df, dtype=float))

        # Location CV and prediction
        # best_contamination = cv_location_contamination(cv_df, cv_labels, rob_mean, rob_precision)

        test_label_vc = test_labels.value_counts()
        ones = test_label_vc.get(1)
        zeros = test_label_vc.get(0)
        best_contamination = ones/(ones + zeros)

        pred_label = predict_by_location_contamination(testing_df, rob_mean, rob_precision, best_contamination)
        l_f1 = f1_score(actual_labels, pred_label)
        print('%s - predict_by_location_contamination - F1: %f' % (files_list[dataset].name, l_f1), file=result_file)
        result_file.flush()

        # Skewness CV and prediction
        # best_contamination = cv_skewness_contamination(cv_df, cv_labels, rob_skew, rob_precision)
        pred_label = predict_by_skewness_contamination(testing_df, rob_precision, rob_skew, best_contamination)
        s_f1 = f1_score(actual_labels, pred_label)
        print('%s - predict_by_skewness_contamination - F1: %f' % (files_list[dataset].name, s_f1), file=result_file)
        result_file.flush()

        # Kurtosis CV and prediction
        # best_contamination = cv_kurtosis_contamination(cv_df, cv_labels, rob_kurt, rob_precision)
        pred_label = predict_by_kurtosis_contamination(testing_df, rob_precision, rob_kurt, best_contamination)
        k_f1 = f1_score(actual_labels, pred_label)
        print('%s - predict_by_kurtosis_contamination - F1: %f' % (files_list[dataset].name, k_f1), file=result_file)
        result_file.flush()

        # # test ROBPCA-AO from saved files
        # robpca_result_file = '../output/ctu_13/results/agg_robpca/robpca_k2_%s_test_df' % file_name
        # # print(robpca_result_file)
        # if os.path.isfile(robpca_result_file):
        #     # print(robpca_result_file)
        #     robpca_test_pred = pd.read_csv(robpca_result_file, header=None)
        #     robpca_test_pred = robpca_test_pred[0]
        #
        #     robpca_test_pred[robpca_test_pred == True] = -1
        #     robpca_test_pred[robpca_test_pred == False] = 0
        #     robpca_test_pred[robpca_test_pred == -1] = 1
        #     print('%s - robpca_k2 - F1: %f' % (files_list[dataset].name, f1_score(test_labels, robpca_test_pred)))

    print("--- Tempo de execução %s segundos ---" % (time.time() - start_time), file=result_file)
result_file.close()