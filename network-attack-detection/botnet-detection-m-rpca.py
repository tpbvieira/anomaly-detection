# coding=utf-8
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from moment_rpca import fit, md_rpca_prediction, sd_rpca_prediction, kd_rpca_prediction
from botnet_detection_utils import data_splitting_50_33, ctu13_data_cleasing, ctu13_raw_column_types
warnings.filterwarnings("ignore")

col_list = ['State', 'dTos', 'Dport', 'Sport', 'TotPkts', 'TotBytes', 'SrcBytes']

start_time = time.time()
raw_path = os.path.join('data/ctu_13/raw_clean_pkl/')
raw_directory = os.fsencode(raw_path)
pkl_path = os.path.join('data/ctu_13/raw_clean_pkl/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)
test_robpca_path = 'data/ctu_13/raw_clean_test_robpca_csv/'

m_f1_list = []
s_f1_list = []
k_f1_list = []
# for each scenario
for sample_file in file_list:

    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(pkl_file_path):
        print("## Sample File: ", pkl_file_path)
        df = pd.read_pickle(pkl_file_path)
    else:  # load raw file and save clean data into pickles
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        print("## Sample File: ", raw_file_path)
        raw_df = pd.read_csv(raw_file_path, header=0, dtype=ctu13_raw_column_types)
        df = ctu13_data_cleasing(raw_df)
        df.to_pickle(pkl_file_path)
    gc.collect()

    # data splitting
    norm_train_df, test_df, test_label_df = data_splitting_50_33(df, col_list)

    test_robpca_file_path = '%s/33_%s' % (test_robpca_path, sample_file)
    print(test_robpca_file_path)
    if not os.path.isfile(test_robpca_file_path):
        test_label_robpca_file_path = '%s/label_33_%s' % (test_robpca_path, sample_file)
        test_df.to_csv(test_robpca_file_path)
        test_label_df.to_csv(test_label_robpca_file_path)

    # # Train
    # L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(
    #     np.array(norm_train_df, dtype=float))
    #
    # # Cross-Validation for best_contamination
    # test_label_vc = test_label_df.value_counts()
    # ones = test_label_vc.get(1)
    # if ones == 0:
    #     continue
    # zeros = test_label_vc.get(0)
    # best_contamination = ones/(ones + zeros)
    # if best_contamination > 0.5:
    #     best_contamination = 0.5
    # print('### Cross-Validation. Contamination:', best_contamination)
    #
    # # Testing md-rpca
    # md_pred_label = md_rpca_prediction(test_df, rob_mean, rob_precision, best_contamination)
    # m_f1 = f1_score(test_label_df, md_pred_label)
    # m_f1_list.append(m_f1)
    # print('%s - md_rpca_prediction - F1: %f' % (sample_file, m_f1))
    #
    # # Testing sd-rpca
    # sd_pred_label = sd_rpca_prediction(test_df, rob_skew, rob_precision, best_contamination)
    # s_f1 = f1_score(test_label_df, sd_pred_label)
    # s_f1_list.append(s_f1)
    # print('%s - sd_rpca_prediction - F1: %f' % (sample_file, s_f1))
    #
    # # Testing kd-rpca
    # kd_pred_label = kd_rpca_prediction(test_df, rob_kurt, rob_precision, best_contamination)
    # k_f1 = f1_score(test_label_df, kd_pred_label)
    # k_f1_list.append(k_f1)
    # print('%s - kd_rpca_prediction - F1: %f' % (sample_file, k_f1))

print('l_mean:', np.mean(m_f1_list))
print('s_mean:', np.mean(s_f1_list))
print('k_mean:', np.mean(k_f1_list))
print("--- %s seconds ---" % (time.time() - start_time))
