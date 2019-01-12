# coding=utf-8
import pandas as pd
import numpy as np
import os, gc, time, warnings
from scipy.stats import multivariate_normal
from sklearn import preprocessing, mixture
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, RobustScaler
warnings.filterwarnings(action='once')


def selectThresholdByCV(probs, labels):
    # select best epsilon (threshold)

    # initialize
    best_epsilon = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)

    for epsilon in epsilons:
        predictions = (probs < epsilon)

        f1 = f1_score(labels, predictions, average="binary")
        Recall = recall_score(labels, predictions, average="binary")
        Precision = precision_score(labels, predictions, average="binary")

        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
            best_precision = Precision
            best_recall = Recall

    return best_f1, best_epsilon


def print_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average="binary")
    Recall = recall_score(y_test, y_predic, average="binary")
    Precision = precision_score(y_test, y_predic, average="binary")
    print('\tF1 Score: ', f1, ', Recall: ', Recall, ', Precision: ,', Precision)


def get_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average="binary")
    Recall = recall_score(y_test, y_predic, average="binary")
    Precision = precision_score(y_test, y_predic, average="binary")

    return f1, Recall, Precision


def model_order_selection(data, max_components):
    bic = []
    lowest_bic = np.infty
    n_components_range = range(1, max_components)
    cov_types = ['spherical', 'tied', 'diag', 'full']

    for cov_type in cov_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cov_type)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_n_components = n_components
                best_cov_type = cov_type

    return best_n_components, best_cov_type


def data_splitting(df, drop_features):
    # Data splitting

    # drop non discriminant features
    # df.drop(drop_features, axis=1, inplace=True)
    gc.collect()

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)  # total number of anomalous flows
    anom_train_end = anom_len // 2  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1  # 50% of anomalous for testing
    norm_len = len(df_l0)  # total number of normal flows
    norm_train_end = (norm_len * 60) // 100  # 60% of normal for training
    norm_cv_start = norm_train_end + 1  # 20% of normal for cross validation
    norm_cv_end = (norm_len * 80) // 100  # 20% of normal for cross validation
    norm_test_start = norm_cv_end + 1  # 20% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]  # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]  # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]  # 60% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]  # 20% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 20% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # save labels and drop labels from data
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]
    norm_train_df = norm_train_df.drop(labels=["Label"], axis=1)
    cv_df = cv_df.drop(labels=["Label"], axis=1)
    test_df = test_df.drop(labels=["Label"], axis=1)

    gc.collect()

    return norm_train_df, cv_df, test_df, cv_label, test_label


start_time = time.time()


# features
column_types = {
    'StartTime': 'str',
    'Dur': 'float32',
    'Proto': 'str',
    'SrcAddr': 'str',
    'Sport': 'str',
    'Dir': 'str',
    'DstAddr': 'str',
    'Dport': 'str',
    'State': 'str',
    'sTos': 'float16',
    'dTos': 'float16',
    'TotPkts': 'uint32',
    'TotBytes': 'uint32',
    'SrcBytes': 'uint32',
    'Label': 'str'}

# feature selection
drop_features = {
    'drop_features00': []
    # 'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'Proto', 'TotBytes', 'SrcBytes'],
    # 'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'TotBytes', 'SrcBytes'],
    # 'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'Proto', 'SrcBytes'],
    # 'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
}

# raw files
raw_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/network-attack-detection/BinetflowTrainer-master/pkl/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)
print("### Directory: ", raw_directory)
print("### Files: ", raw_files)

# pickle files - have the same names but different directory
pkl_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/network-attack-detection/BinetflowTrainer-master/pkl/')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_features.items():

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
        norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, [])

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