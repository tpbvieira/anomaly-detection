# coding=utf-8
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from moment_rpca import fit, md_rpca_prediction, sd_rpca_prediction, kd_rpca_prediction
from botnet_detection_utils import data_splitting_50_25, ctu13_data_cleasing, ctu13_raw_column_types
warnings.filterwarnings("ignore")

result_file_path = 'm_rpca_cv.txt'
result_file = open(result_file_path, 'w') # 'w' = clear all and write
col_list = ['State', 'sTos', 'dTos', 'Proto', 'Dport', 'Sport', 'Dir', 'Dur', 'TotPkts', 'TotBytes', 'SrcBytes',
            'PktsRate', 'BytesRate', 'MeanPktsRate']

raw_path = os.path.join('data/ctu_13/raw_clean_pkl/')
raw_directory = os.fsencode(raw_path)
pkl_path = os.path.join('data/ctu_13/raw_clean_pkl/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

best_global_f1 = 0
best_global_cols = None
best_global_alg = None
start_time = time.time()
while len(col_list) > 2:
    best_f1 = 0
    best_cols = None
    best_alg = None
    for col in col_list:

        col_list_md_f1 = []
        col_list_sd_f1 = []
        col_list_kd_f1 = []
        n_col_list = col_list.copy()
        n_col_list.remove(col)

        for sample_file in file_list:

            # read pickle file with pandas or...
            pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
            if os.path.isfile(pkl_file_path):
                # print("## Sample File: ", pkl_file_path, file=result_file)
                df = pd.read_pickle(pkl_file_path)
            else:  # load raw file and save clean data into pickles
                raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
                # print("## Sample File: ", raw_file_path, file=result_file)
                raw_df = pd.read_csv(raw_file_path, header=0, dtype=ctu13_raw_column_types)
                df = ctu13_data_cleasing(raw_df)
                df.to_pickle(pkl_file_path)
            gc.collect()

            # data splitting
            norm_train_df, test_df, test_label_df = data_splitting_50_25(df, n_col_list)

            # Train
            L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(
                np.array(norm_train_df, dtype=float))

            # Cross-Validation for best_contamination
            test_label_vc = test_label_df.value_counts()
            ones = test_label_vc.get(1)
            if ones == 0:
                continue
            zeros = test_label_vc.get(0)
            best_contamination = ones / (ones + zeros)
            if best_contamination > 0.5:
                best_contamination = 0.5
            # print('### Cross-Validation. Contamination:', best_contamination, file=result_file)

            # # Testing md-rpca
            # md_pred_label = md_rpca_prediction(test_df, rob_mean, rob_precision, best_contamination)
            # sample_md_f1 = f1_score(test_label_df, md_pred_label)
            # col_list_md_f1.append(sample_md_f1)
            # # print('%s - md_rpca_prediction - F1: %f' % (sample_file, m_f1), file=result_file)
            # result_file.flush()

            # Testing sd-rpca
            sd_pred_label = sd_rpca_prediction(test_df, rob_skew, rob_precision, best_contamination)
            sample_sd_f1 = f1_score(test_label_df, sd_pred_label)
            col_list_sd_f1.append(sample_sd_f1)
            # print('%s - sd_rpca_prediction - F1: %f' % (sample_file, s_f1), file=result_file)
            result_file.flush()

            # # Testing kd-rpca
            # kd_pred_label = kd_rpca_prediction(test_df, rob_kurt, rob_precision, best_contamination)
            # sample_kd_f1 = f1_score(test_label_df, kd_pred_label)
            # col_list_kd_f1.append(sample_kd_f1)
            # # print('%s - kd_rpca_prediction - F1: %f' % (sample_file, k_f1), file=result_file)
            # result_file.flush()

        # col_list_md_f1 = np.mean(col_list_md_f1)
        # if col_list_md_f1 > best_f1:
        #     best_f1 = col_list_md_f1
        #     best_cols = n_col_list.copy()
        #     best_alg = 'md-rpca'

        col_list_sd_f1 = np.mean(col_list_sd_f1)
        if col_list_sd_f1 > best_f1:
            best_f1 = col_list_sd_f1
            best_cols = n_col_list.copy()
            best_alg = 'sd-rpca'

        # col_list_kd_f1 = np.mean(col_list_kd_f1)
        # if col_list_kd_f1 > best_f1:
        #     best_f1 = col_list_kd_f1
        #     best_cols = n_col_list.copy()
        #     best_alg = 'kd-rpca'

        print(n_col_list, np.mean(col_list_sd_f1), file=result_file)
        result_file.flush()

    print('### :', best_f1, best_alg, best_cols, file=result_file)
    result_file.flush()

    col_list = best_cols
    if best_f1 >= best_global_f1:
        best_global_f1 = best_f1
        best_global_cols = best_cols.copy()
        best_global_alg = best_alg
        print('### Improved_Global_Mean:', best_global_f1, best_global_alg, best_global_cols, file=result_file)

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