# coding=utf-8
import time
import numpy as np
from sklearn.metrics import f1_score
from algorithms import fit, \
    cv_location_contamination, cv_location_threshold, cv_skewness_contamination, cv_kurtosis_contamination, \
    cv_skewness_threshold, cv_kurtosis_threshold, \
    predict_by_location_contamination, predict_by_location_centered_contamination, predict_by_location_threshold, \
    predict_by_skewness_contamination, predict_by_skewness_centered_contamination, predict_by_skewness_threshold, predict_by_skewness_centered_threshold, \
    predict_by_kurtosis_contamination, predict_by_kurtosis_centered_contamination, predict_by_kurtosis_threshold, predict_by_kurtosis_centered_threshold
from data_processing import get_dataset, split_dataset
import warnings
warnings.filterwarnings("ignore")

result_file_path = 'moment_rpca.txt'
result_file = open(result_file_path, 'w') # 'w' = clear all and write

start_time = time.time()
prefixes = ['0.15']
for prefix in prefixes:
    print('### Dataset Version: %s' % prefix, file=result_file)
    result_file.flush()
    dataset_list, files_list = get_dataset(prefix)
    pre_processed_dataset = split_dataset(dataset_list)

    col_list = ['n_dports>1024', 'flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address',
                'n_sports<1024', 'n_sports>1024', 'n_conn', 'n_s_na_p_address', 'n_udp', 'n_icmp', 'n_d_na_p_address',
                'n_d_a_p_address', 'n_s_c_p_address', 'n_d_c_p_address', 'normal_flow_count', 'n_dports<1024',
                'n_d_b_p_address', 'n_tcp']

    best_global_f1 = 0
    best_global_cols = None
    best_global_alg = None
    while len(col_list) > 2:
        best_f1 = 0
        best_cols = None
        best_alg = None
        for col in col_list:
            l_f1_list = []
            s_f1_list = []
            k_f1_list = []
            n_col_list = col_list.copy()
            n_col_list.remove(col)
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

                training_df = raw_train_df[n_col_list]
                cv_df = raw_cv_df[n_col_list]
                testing_df = raw_test_df[n_col_list]

                # Training
                L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(np.array(training_df, dtype=float))

                # # Location CV and prediction
                # best_contamination = cv_location_contamination(cv_df, cv_labels, rob_mean, rob_precision)
                # pred_label = predict_by_location_contamination(testing_df, rob_mean, rob_precision, best_contamination)
                # l_f1 = f1_score(actual_labels, pred_label)
                # l_f1_list.append(l_f1)
                # result_file.flush()

                # Skewness CV and prediction
                best_contamination = cv_skewness_contamination(cv_df, cv_labels, rob_skew, rob_precision)
                pred_label = predict_by_skewness_contamination(testing_df, rob_precision, rob_skew, best_contamination)
                s_f1 = f1_score(actual_labels, pred_label)
                s_f1_list.append(s_f1)
                result_file.flush()

                # # Kurtosis CV and prediction
                # best_contamination = cv_kurtosis_contamination(cv_df, cv_labels, rob_kurt, rob_precision)
                # pred_label = predict_by_kurtosis_contamination(testing_df, rob_precision, rob_kurt, best_contamination)
                # k_f1 = f1_score(actual_labels, pred_label)
                # k_f1_list.append(k_f1)
                # result_file.flush()

            # l_f1 = np.mean(l_f1_list)
            # if l_f1 > best_f1:
            #     best_f1 = np.mean(l_f1_list)
            #     best_cols = n_col_list.copy()
            #     best_alg = 'l-rpca'
            s_f1 = np.mean(s_f1_list)
            if s_f1 > best_f1:
                best_f1 = np.mean(s_f1_list)
                best_cols = n_col_list.copy()
                best_alg = 's-rpca'
            # k_f1_list = np.mean(k_f1_list)
            # if k_f1_list > best_f1:
            #     best_f1 = np.mean(k_f1_list)
            #     best_cols = n_col_list.copy()
            #     best_alg = 'k-rpca'

            print(n_col_list, s_f1, file=result_file)
        print('### :', prefix, best_f1, best_alg, best_cols, file=result_file)

        col_list = best_cols
        if best_f1 >= best_global_f1:
            best_global_f1 = best_f1
            best_global_cols = best_cols.copy()
            print('### Improved_Global_Mean:',prefix, best_global_f1, best_alg, best_global_cols, file=result_file)

        # # test ROBPCA-AO from saved files
        # robpca_result_file = '/home/thiago/dev/anomaly-detection/network-attack-detection/output/ctu_13/results/m-rpca_r/robpca_k19_%s_test_df' % file_name
        # if os.path.isfile(robpca_result_file):
        #     # print(robpca_result_file)
        #     robpca_test_pred = pd.read_csv(robpca_result_file, header=None)
        #     robpca_test_pred = robpca_test_pred[0]
        #
        #     robpca_test_pred[robpca_test_pred == True] = -1
        #     robpca_test_pred[robpca_test_pred == False] = 0
        #     robpca_test_pred[robpca_test_pred == -1] = 1
        #     print('%s - robpca_k19_%s_test_df - F1: %f' % (
        #         files_list[dataset].name, file_name, f1_score(test_labels, robpca_test_pred)))

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