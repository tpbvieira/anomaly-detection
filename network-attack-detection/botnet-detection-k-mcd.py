# coding=utf-8
import os, gc, time, warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from k_mcd_outlier_detection import MEllipticEnvelope
from botnet_detection_utils import data_splitting, drop_agg_features, get_classification_report, data_cleasing, column_types
warnings.filterwarnings("ignore")


def getBestBySemiSupervCV(t_normal_df, t_cv_df, t_cv_label):
    m_cv_label = t_cv_label.astype(np.int8)

    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_precision = -1
    m_best_recall = -1

    # configure GridSearchCV
    for m_contamination in np.linspace(0.01, 0.2, 10):
        m_ell_model = MEllipticEnvelope(contamination=m_contamination)
        m_ell_model.fit(t_normal_df)
        m_pred = m_ell_model.kurtosis_prediction(t_cv_df)

        m_f1 = f1_score(m_cv_label, m_pred, average="binary")
        m_recall = recall_score(m_cv_label, m_pred, average="binary")
        m_precision = precision_score(m_cv_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_model = m_ell_model
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = m_precision
            m_best_recall = m_recall

    return m_best_model, m_best_contamination, m_best_f1, m_best_precision, m_best_recall


def getBestBySemiSupervCVWithCI(t_normal_df, t_cv_df, t_cv_label, n_it):
    # initialize
    m_cv_label = t_cv_label.astype(np.int8)
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1

    for m_contamination in np.linspace(0.01, 0.2, 10):
        t_f1 = []
        for i in range(n_it):
            m_ell_model = MEllipticEnvelope(contamination=m_contamination)
            m_ell_model.fit(t_normal_df)
            m_pred = m_ell_model.kurtosis_prediction(t_cv_df)
            t_f1.append(f1_score(m_cv_label, m_pred, average="binary"))

        m_f1 = np.median(t_f1)

        if m_f1 > m_best_f1:
            m_best_model = m_ell_model
            m_best_contamination = m_contamination
            m_best_f1 = m_f1

    return m_best_model, m_best_contamination, m_best_f1

start_time = time.time()

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum/')
raw_directory = os.fsencode(raw_path)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

it = 10

# for each feature set
for features_key, value in drop_agg_features.items():

    # for each file/case
    for sample_file in file_list:
        result_file = "k-mcd_%d_%s" % (it, sample_file.decode('utf-8'))
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
            norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_agg_features[features_key])

            # Cross-Validation and model selection
            ell_model, best_contamination, best_f1 = getBestBySemiSupervCVWithCI(norm_train_df, cv_df, cv_label, it)
            print('###[k-mcd][', features_key, '] Cross-Validation. Contamination:', best_contamination,', F1:', best_f1)

            # Test
            m_f1 = []
            m_pr = []
            m_re = []
            test_label = test_label.astype(np.int8)
            for i in range(it):
                pred_test_label = ell_model.kurtosis_prediction(test_df)
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, pred_test_label)
                m_f1.append(t_f1)
                m_pr.append(t_Precision)
                m_re.append(t_Recall)

            # print results
            print('###[k-mcd][', features_key, '] Test. mF1:', np.median(m_f1), ', mRecall:', np.median(m_re), ', mPrecision:', np.median(m_pr))
            df = pd.DataFrame([m_f1, m_re, m_pr])
            df.to_pickle(result_file)

print("--- %s seconds ---" % (time.time() - start_time))