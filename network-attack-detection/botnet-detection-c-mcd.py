# coding=utf-8
import os, gc, time, warnings, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import shuffle
from k_mcd_outlier_detection import MEllipticEnvelope
from botnet_detection_utils import data_splitting, unsupervised_data_splitting, drop_agg_features, get_classification_report, data_cleasing, column_types
warnings.filterwarnings("ignore")


def semiSupervisedCV(t_normal_df, t_cv_df, t_cv_label, n_it):
    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1

    # suffle cv data
    t_cv_df['Labels'] = t_cv_label
    t_cv_df = shuffle(t_cv_df)
    t_cv_label = t_cv_df['Labels']
    m_cv_label = t_cv_label.astype(np.int8)
    t_cv_df = t_cv_df.drop('Labels', 1)

    for m_contamination in np.linspace(0.1, 0.4, 10):

        for i in range(n_it):
            # fit
            m_ell_model = MEllipticEnvelope(contamination=m_contamination)
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

            # kurtosis prediction
            k_pred = m_ell_model.kurtosis_prediction(t_cv_df)
            k_f1 = f1_score(m_cv_label, k_pred, average="binary")
            if k_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = k_f1

            # skewness prediction
            s_pred = m_ell_model.skewness_prediction(t_cv_df)
            s_f1 = f1_score(m_cv_label, s_pred, average="binary")
            if s_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = s_f1

    return m_best_model, m_best_contamination, m_best_f1


def unsupervisedCV(t_cv_df, t_cv_label, n_it):
    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1

    # suffle cv data
    t_cv_df['Labels'] = t_cv_label
    t_cv_df = shuffle(t_cv_df)
    t_cv_label = t_cv_df['Labels']
    m_cv_label = t_cv_label.astype(np.int8)
    t_cv_df = t_cv_df.drop('Labels', 1)

    for m_contamination in np.linspace(0.1, 0.8, 10):

        for i in range(n_it):
            # fit
            m_ell_model = MEllipticEnvelope(contamination=m_contamination)
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

            # kurtosis prediction
            k_pred = m_ell_model.kurtosis_prediction(t_cv_df)
            k_f1 = f1_score(m_cv_label, k_pred, average="binary")
            if k_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = k_f1

            # skewness prediction
            s_pred = m_ell_model.skewness_prediction(t_cv_df)
            s_f1 = f1_score(m_cv_label, s_pred, average="binary")
            if s_f1 > m_best_f1:
                m_best_model = m_ell_model
                m_best_contamination = m_contamination
                m_best_f1 = s_f1

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
            # norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_agg_features[features_key])
            train_df, train_label_df, test_df, test_label_df = unsupervised_data_splitting(df, drop_agg_features[features_key])

            c_mcd_result_dict = {}
            for i in range(it):
                # Cross-Validation and model selection
                # ell_model, best_contamination, best_f1 = semiSupervisedCV(norm_train_df, cv_df, cv_label, 1)
                ell_model, best_contamination, best_f1 = unsupervisedCV(train_df, train_label_df, 1)
                print('###[c-mcd][', features_key, '] Cross-Validation. Contamination:', best_contamination,', F1:', best_f1)

                test_label = test_label_df.astype(np.int8)

                # MCD Prediction Test
                mcd_pred_label = ell_model.mcd_prediction(test_df)
                mcd_pred_dist_ = ell_model.prediction_dist_
                mcd_pred_label[mcd_pred_label == 1] = 0
                mcd_pred_label[mcd_pred_label == -1] = 1
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, mcd_pred_label)
                print('###[mcd][', features_key, '] Test. F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)

                # K-MCD Prediction Test
                k_pred_label = ell_model.kurtosis_prediction(test_df)
                k_pred_dist_ = ell_model.prediction_dist_
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, k_pred_label)
                print('###[k-mcd][', features_key, '] Test. F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)

                # S-MCD Prediction Test
                s_pred_label = ell_model.skewness_prediction(test_df)
                s_pred_dist_ = ell_model.prediction_dist_
                t_f1, t_Recall, t_Precision = get_classification_report(test_label, s_pred_label)
                print('###[s-mcd][', features_key, '] Test. F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)

            #     c_mcd_pred = {
            #         "best_contamination": best_contamination,
            #         "train_f1": best_f1,
            #         "mcd_pred_dist_": mcd_pred_dist_,
            #         "k_pred_dist_": k_pred_dist_,
            #         "s_pred_dist_": s_pred_dist_,
            #         "mcd_pred_label": mcd_pred_label,
            #         "k_pred_label": k_pred_label,
            #         "s_pred_label": s_pred_label,
            #         "test_label": test_label
            #     }
            #     c_mcd_result_dict[i] = c_mcd_pred
            #
            # # write python dict to a file
            # output = open(result_file, 'wb')
            # pickle.dump(c_mcd_result_dict, output)
            # output.close()

print("--- %s seconds ---" % (time.time() - start_time))