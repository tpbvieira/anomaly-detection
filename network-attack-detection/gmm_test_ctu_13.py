# coding=utf-8
import pandas as pd
import numpy as np
import os, gc, time, warnings
from sklearn import mixture
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from botnet_detection_utils import data_splitting, drop_agg_features, print_classification_report
warnings.filterwarnings(action='once')


def selectThresholdByCV(probs, labels):
    # select best epsilon (threshold)

    # initialize
    best_epsilon = 0
    best_f1 = 0

    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)

    for epsilon in epsilons:
        predictions = (probs < epsilon)
        f1 = f1_score(labels, predictions, average="binary")

        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1

    return best_f1, best_epsilon


start_time = time.time()

# raw files
raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum_fast')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)
print("### Directory: ", raw_directory)
print("### Files: ", raw_files)

# pickle files - have the same names but different directory
pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum_fast')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_agg_features.items():

    # initialize labels
    gmm_cv_label = []
    gmm_pred_cv_label = []
    gmm_test_label = []
    gmm_pred_test_label = []

    # for each file
    for sample_file in raw_files:
        pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_file_path):
            print("### Sample File: ", pkl_file_path)
            df = pd.read_pickle(pkl_file_path)
        else:
            print("### Sample File: ", raw_file_path)
            raw_df = pd.read_csv(raw_file_path, header=0, dtype=column_types)
            df = data_cleasing(raw_df)
        gc.collect()

        # data splitting
        norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_agg_features[features_key])

        # Scaler: raw, standardization (zero mean and unitary variance) or robust scaler
        scaler = 'RobustScaler'
        norm_train_values = RobustScaler().fit_transform(norm_train_df.values)
        norm_train_df = pd.DataFrame(norm_train_values, index=norm_train_df.index, columns=norm_train_df.columns)
        gc.collect()
        cv_values = RobustScaler().fit_transform(cv_df.values)
        cv_df = pd.DataFrame(cv_values, index=cv_df.index, columns=cv_df.columns)
        gc.collect()
        test_values = RobustScaler().fit_transform(test_df.values)
        test_df = pd.DataFrame(test_values, index=test_df.index, columns=test_df.columns)
        gc.collect()

        try:
            # [GMM] Fit a Gaussian Mixture Model
            # best_n_components, best_cov_type = model_order_selection(norm_train_df, len(norm_train_df.columns))
            gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')
            gmm.fit(norm_train_df)

            # [GMM] Cross Validation and threshold selection
            p_cv = gmm.score_samples(cv_df)

            # [GMM] Cross Validation
            fscore, epsilon = selectThresholdByCV(p_cv, cv_label)
            pred_cv_label = (p_cv < epsilon)
            gmm_cv_label.extend(cv_label.astype(int))
            gmm_pred_cv_label.extend(pred_cv_label.astype(int))
            print('###', features_key, scaler, '[GMM]', 'Classification report for Cross Validation dataset')
            print_classification_report(cv_label, pred_cv_label)

            # [GMM] Test
            p_test = gmm.score_samples(test_df)
            pred_label = (p_test < epsilon)
            gmm_test_label.extend(test_label.astype(int))
            gmm_pred_test_label.extend(pred_label.astype(int))
            print('###', features_key, scaler, '[GMM]', 'Classification report for Test dataset')
            print_classification_report(test_label, pred_label)
        except:
            print("### [GMM] Error!!")

    # [GMM] print results
    print('###', features_key, scaler, '[GMM]', 'Total Classification report for Cross Validation dataset')
    print_classification_report(gmm_cv_label, gmm_pred_cv_label)
    print('###', features_key, scaler, '[GMM]', 'Total Classification report for Test dataset')
    print_classification_report(gmm_test_label, gmm_pred_test_label)

print("--- %s seconds ---" % (time.time() - start_time))