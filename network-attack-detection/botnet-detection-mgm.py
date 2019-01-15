# coding=utf-8
import pandas as pd
import numpy as np
import os, gc, warnings, time
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, recall_score, precision_score
from botnet_detection_utils import data_splitting, drop_agg_features, data_cleasing, column_types
warnings.filterwarnings(action='once')


def estimate_gaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariate_gaussian(dataset, mu, sigma):
    mg_model = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return mg_model.logpdf(dataset)


def get_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average="binary")
    Recall = recall_score(y_test, y_predic, average="binary")
    Precision = precision_score(y_test, y_predic, average="binary")

    return f1, Recall, Precision


def selectThresholdByCV(pred, labels):

    # initialize
    best_epsilon = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    t_df = pd.DataFrame({'pred': pred, 'labels': labels.values})
    anomalies = t_df[t_df['labels'] == 1]
    min_prob = min(anomalies['pred'])
    max_prob = max(anomalies['pred'])
    stepsize = (max_prob - min_prob) / 100  # divided by the expected number of steps
    epsilons = np.arange(min(pred), max(pred), stepsize)

    for epsilon in epsilons:
        predictions = (pred < epsilon)

        f1 = f1_score(labels, predictions, average="binary")
        Recall = recall_score(labels, predictions, average="binary")
        Precision = precision_score(labels, predictions, average="binary")

        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
            best_precision = Precision
            best_recall = Recall

    return best_epsilon, best_f1, best_precision, best_recall


start_time = time.time()

# raw files
raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum_fast')
raw_directory = os.fsencode(raw_path)

# pickle files - have the same names but different directory
pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum_fast')
pkl_directory = os.fsencode(pkl_path)
pkl_files = os.listdir(pkl_directory)
print("### Directory: ", pkl_directory)

# for each feature set
for features_key, value in drop_agg_features.items():

    # initialize labels
    mgm_cv_label = []
    mgm_pred_cv_label = []
    mgm_test_label = []
    mgm_pred_test_label = []
    scaler = ''

    # for each file
    for sample_file in pkl_files:
        pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_file_path):
            print("### Sample File: ", pkl_file_path)
            data = pd.read_pickle(pkl_file_path)
        else:
            print("### Sample File: ", raw_file_path)
            raw_data = pd.read_csv(raw_file_path, header=0, dtype=column_types)
            data = data_cleasing(raw_data)
        gc.collect()

        # data splitting
        norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(data, drop_agg_features[features_key])

        # Scaler: raw, standardization (zero mean and unitary variance) or robust scaler
        scaler = 'Raw'
        #                   norm_train_values = RobustScaler().fit_transform(norm_train_df.values)
        #                   norm_train_df = pd.DataFrame(norm_train_values, index=norm_train_df.index, columns=norm_train_df.columns)
        #                   gc.collect()
        #                   cv_values = RobustScaler().fit_transform(cv_df.values)
        #                   cv_df = pd.DataFrame(cv_values, index=cv_df.index, columns=cv_df.columns)
        #                   gc.collect()
        #                   test_values = RobustScaler().fit_transform(test_df.values)
        #                   test_df = pd.DataFrame(test_values, index=test_df.index, columns=test_df.columns)
        #                   gc.collect()

        try:
            # trainning - estimate the mean vector and the covariance matrix from the normal data
            mu, sigma = estimate_gaussian(norm_train_df)

            # Cross-Validation
            p_cv = multivariate_gaussian(cv_df, mu, sigma)
            best_epsilon, best_f1, best_precision, best_recall = selectThresholdByCV(p_cv, cv_label)
            pred_cv_label = (p_cv < best_epsilon)
            print('### [MGM][', features_key, '][', scaler, '] Cross-Validation. Epsilon: ', best_epsilon, ', F1: ', best_f1, ', Precision: ', best_precision, ', Recall: ', best_recall)

            # [MGM] Test
            p_test = multivariate_gaussian(test_df, mu, sigma)
            pred_test_label = (p_test < best_epsilon)
            f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
            print('### [MGM][', features_key, '][', scaler, '] Test. F1: ', f1, ', Precision: ', Precision, ', Recall: ', Recall)

            mgm_cv_label.extend(cv_label.astype(int))
            mgm_pred_cv_label.extend(pred_cv_label.astype(int))
            mgm_test_label.extend(test_label.astype(int))
            mgm_pred_test_label.extend(pred_test_label.astype(int))

        except Exception as e:
            print("### [MGM] Error: ", str(e))

    # [MGM] print results
    f1, Recall, Precision = get_classification_report(mgm_cv_label, mgm_pred_cv_label)
    print('### [MGM][', features_key, '][', scaler, '] Cross-Validation of all data. F1: ', f1, ', Precision: ', Precision, ', Recall: ', Recall)

    f1, Recall, Precision = get_classification_report(mgm_test_label, mgm_pred_test_label)
    print('### [MGM][', features_key, '][', scaler, '] Test of all data. F1: ', f1, ', Precision: ', Precision, ', Recall: ', Recall)

print("--- %s seconds ---" % (time.time() - start_time))