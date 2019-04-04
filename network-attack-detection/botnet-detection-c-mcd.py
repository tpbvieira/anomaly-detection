# coding=utf-8
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from moment_anomaly_detector import MomentAnomalyDetector
from botnet_detection_utils import data_splitting, drop_agg_features, get_classification_report, data_cleasing, column_types
warnings.filterwarnings("ignore")


def semiSupervisedCV(t_normal_df, t_cv_df, t_cv_label, n_it):
    # initialize
    m_best_model = MomentAnomalyDetector()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_alg = ''

    # suffle cv data
    t_cv_df['Labels'] = t_cv_label
    t_cv_df = shuffle(t_cv_df)
    t_cv_label = t_cv_df['Labels']
    m_cv_label = t_cv_label.astype(np.int8)
    t_cv_df = t_cv_df.drop('Labels', 1)

    for m_contamination in np.linspace(0.1, 0.4, 10):

        for j in range(n_it):
            # fit
            m_ell_model = MomentAnomalyDetector(contamination=m_contamination)
            m_ell_model.fit(t_normal_df)

            # mcd prediction
            m_pred = m_ell_model.mcd_prediction(t_cv_df)
            m_pred[m_pred == 1] = 0
            m_pred[m_pred == -1] = 1
            mcd_f1 = f1_score(m_cv_label, m_pred, average="binary")
            if mcd_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = mcd_f1
                m_best_alg = 'mcd'

            # kurtosis prediction
            k_pred = m_ell_model.kurtosis_prediction(t_cv_df)
            k_f1 = f1_score(m_cv_label, k_pred, average="binary")
            if k_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = k_f1
                m_best_alg = 'k-mcd'

            # skewness prediction
            s_pred = m_ell_model.skewness_prediction(t_cv_df)
            s_f1 = f1_score(m_cv_label, s_pred, average="binary")
            if s_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = s_f1
                m_best_alg = 's-mcd'

    return m_best_model, m_best_contamination, m_best_f1, m_best_alg


def unsupervisedCV(t_cv_df, t_cv_label, n_it):
    # initialize
    m_best_model = MomentAnomalyDetector()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_alg = ''

    # suffle cv data
    t_cv_df['Labels'] = t_cv_label
    t_cv_df = shuffle(t_cv_df)
    t_cv_label = t_cv_df['Labels']
    m_cv_label = t_cv_label.astype(np.int8)
    t_cv_df = t_cv_df.drop('Labels', 1)

    for m_contamination in np.linspace(0.1, 0.8, 10):

        for ite in range(n_it):
            # fit
            m_ell_model = MomentAnomalyDetector(contamination=m_contamination)
            m_ell_model.fit(t_cv_df)

            # mcd prediction
            m_pred = m_ell_model.mcd_prediction(t_cv_df)
            m_pred[m_pred == 1] = 0
            m_pred[m_pred == -1] = 1
            mcd_f1 = f1_score(m_cv_label, m_pred, average="binary")
            if mcd_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = mcd_f1
                m_best_alg = 'mcd'

            # kurtosis prediction
            m_pred_k_label = m_ell_model.kurtosis_prediction(t_cv_df)
            k_f1 = f1_score(m_cv_label, m_pred_k_label, average="binary")
            if k_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = k_f1
                m_best_alg = 'k-mcd'

            # skewness prediction
            m_pred_s_label = m_ell_model.skewness_prediction(t_cv_df)
            s_f1 = f1_score(m_cv_label, m_pred_s_label, average="binary")
            if s_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = s_f1
                m_best_alg = 's-mcd'

    return m_best_model, m_best_contamination, m_best_f1, m_best_alg


raw_path = os.path.join('data/ctu_13/pkl_sum_fast/')
raw_directory = os.fsencode(raw_path)
pkl_path = os.path.join('data/ctu_13/pkl_sum_fast/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)
it = 1

start_time = time.time()
# for each feature set
for features_key, value in drop_agg_features.items():

    # for each file/case
    for sample_file in file_list:

        result_file = "results/pkl_sum_dict/%d/data/c-mcd_%d_%s" % (it, it, sample_file.decode('utf-8'))

        if not os.path.isfile(result_file):

            # read pickle file with pandas or...
            pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
            if os.path.isfile(pkl_file_path):
                print("## Sample File: ", pkl_file_path)
                df = pd.read_pickle(pkl_file_path)
            else:  # load raw file and save clean data into pickles
                raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
                print("## Sample File: ", raw_file_path)
                raw_df = pd.read_csv(raw_file_path, header=0, dtype=column_types)
                df = data_cleasing(raw_df)
                df.to_pickle(pkl_file_path)
            gc.collect()

            # data splitting
            norm_train_df, cv_df, test_df, cv_label_df, test_label_df = data_splitting(df, drop_agg_features[features_key])
            # train_df, train_label_df, test_df, test_label_df = unsupervised_data_splitting(df, drop_agg_features[features_key])

            m_mcd_result_dict = {}
            for i in range(it):

                # Cross-Validation and model selection
                train_best_model, train_best_cont, train_best_f1, train_best_alg = semiSupervisedCV(norm_train_df, cv_df, cv_label_df, 1)
                test_label = test_label_df.astype(np.int8)

                # train_best_model, train_best_cont, train_best_f1, train_best_alg = unsupervisedCV(train_df, train_label_df, 1)
                print('###[m-mcd][', features_key, '] Cross-Validation. Contamination:', train_best_cont,', F1:', train_best_f1,', Alg:', train_best_alg)

                # mcd Prediction Test
                pred_m_label = train_best_model.mcd_prediction(test_df)
                pred_m_dist_ = train_best_model.prediction_dist_
                pred_m_label[pred_m_label == 1] = 0
                pred_m_label[pred_m_label == -1] = 1
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, pred_m_label)
                print('\t###[mcd][', features_key, '] Test. F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)

                # k-mcd Prediction Test
                pred_k_label = train_best_model.kurtosis_prediction(test_df)
                pred_k_dist_ = train_best_model.prediction_dist_
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, pred_k_label)
                print('\t###[k-mcd][', features_key, '] Test. F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)

                # s-mcd Prediction Test
                pred_s_label = train_best_model.skewness_prediction(test_df)
                pred_s_dist_ = train_best_model.prediction_dist_
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, pred_s_label)
                print('\t###[s-mcd][', features_key, '] Test. F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)

                m_mcd_pred = {
                    "train_best_cont": train_best_cont,
                    "train_best_f1": train_best_f1,
                    "train_best_alg": train_best_alg,
                    "pred_m_dist_": pred_m_dist_,
                    "pred_k_dist_": pred_k_dist_,
                    "pred_s_dist_": pred_s_dist_,
                    "pred_m_label": pred_m_label,
                    "pred_k_label": pred_k_label,
                    "pred_s_label": pred_s_label,
                    "test_label": test_label
                }
                m_mcd_result_dict[i] = m_mcd_pred

            # # write python dict to a file
            # output = open(result_file, 'wb')
            # pickle.dump(m_mcd_result_dict, output)
            # output.close()

print("--- %s seconds ---" % (time.time() - start_time))