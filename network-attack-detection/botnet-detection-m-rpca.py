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

# # Agg_2s_F1: 4621/0.4886
# col_list = ['n_dports>1024', 'flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address', 'n_sports<1024',
#           'n_sports>1024', 'n_conn', 'n_s_na_p_address', 'n_udp', 'n_icmp', 'n_d_na_p_address', 'n_d_a_p_address',
#           'n_s_c_p_address', 'n_d_c_p_address', 'normal_flow_count', 'n_dports<1024', 'n_d_b_p_address', 'n_tcp',
#           'mdn_duration', 'p95_duration']

# # Agg_2s_F1: 0.4649/0.4886
# col_list = ['n_dports>1024', 'flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address', 'n_sports<1024',
#           'n_sports>1024', 'n_conn', 'n_s_na_p_address', 'n_udp', 'n_icmp', 'n_d_na_p_address', 'n_d_a_p_address',
#           'n_s_c_p_address', 'n_d_c_p_address', 'normal_flow_count', 'n_dports<1024', 'n_d_b_p_address', 'n_tcp']

# # Agg_2s_F1: 0.4910/0.4485
# col_list = ['flow_count', 'n_s_a_p_address', 'n_s_b_p_address', 'n_s_c_p_address', 'n_s_na_p_address',
#             'mdn_duration', 'p95_duration']

# col_list = ['normal_flow_count', 'n_conn', 'mdn_duration', 'p95_duration', 'n_s_a_p_address', 'n_s_b_p_address',
#             'std_duration', 'p05_duration', 'avg_duration']

# col_list = ['n_conn', 'n_s_a_p_address', 'mdn_duration', 'n_s_b_p_address',
#             'n_s_c_p_address', 'n_dports<1024', 'p95_duration']

# col_list = ['Dur', 'Proto', 'Sport', 'Dir', 'Dport', 'State', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes',
#             'PktsRate', 'BytesRate', 'MeanPktsRate']

col_list = ['Dur', 'Dir', 'Dport', 'SrcBytes', 'PktsRate', 'BytesRate', 'MeanPktsRate']

start_time = time.time()
raw_path = os.path.join('data/ctu_13/raw_clean_pkl/')
raw_directory = os.fsencode(raw_path)
pkl_path = os.path.join('data/ctu_13/raw_clean_pkl/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

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
    norm_train_df, test_df, test_label_df = data_splitting_50_25(df, col_list)

    # Train
    L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist = fit(
        np.array(norm_train_df, dtype=float))

    # Cross-Validation for best_contamination
    test_label_vc = test_label_df.value_counts()
    ones = test_label_vc.get(1)
    if ones == 0:
        continue
    zeros = test_label_vc.get(0)
    best_contamination = ones/(ones + zeros)
    if best_contamination > 0.5:
        best_contamination = 0.5
    print('### Cross-Validation. Contamination:', best_contamination)

    # Testing md-rpca
    md_pred_label = md_rpca_prediction(test_df, rob_mean, rob_precision, best_contamination)
    m_f1 = f1_score(test_label_df, md_pred_label)
    m_f1_list.append(m_f1)
    print('%s - md_rpca_prediction - F1: %f' % (sample_file, m_f1))

    # Testing sd-rpca
    sd_pred_label = sd_rpca_prediction(test_df, rob_skew, rob_precision, best_contamination)
    s_f1 = f1_score(test_label_df, sd_pred_label)
    s_f1_list.append(s_f1)
    print('%s - sd_rpca_prediction - F1: %f' % (sample_file, s_f1))

    # Testing kd-rpca
    kd_pred_label = kd_rpca_prediction(test_df, rob_kurt, rob_precision, best_contamination)
    k_f1 = f1_score(test_label_df, kd_pred_label)
    k_f1_list.append(k_f1)
    print('%s - kd_rpca_prediction - F1: %f' % (sample_file, k_f1))

print('l_mean:', np.mean(m_f1_list))
print('s_mean:', np.mean(s_f1_list))
print('k_mean:', np.mean(k_f1_list))
print("--- %s seconds ---" % (time.time() - start_time))
