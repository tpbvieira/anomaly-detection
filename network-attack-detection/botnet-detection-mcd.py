# coding=utf-8
import os, gc, time, warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.covariance import EllipticEnvelope
warnings.filterwarnings("ignore")


def print_classification_report(y_test, y_predic):
    m_f1 = f1_score(y_test, y_predic, average="binary")
    m_recall = recall_score(y_test, y_predic, average="binary")
    m_precision = precision_score(y_test, y_predic, average="binary")
    print('\tF1 Score: ', m_f1, ', Recall: ', m_recall, ', Precision: ,', m_precision)


def get_classification_report(y_test, y_predic):
    m_f1 = f1_score(y_test, y_predic, average = "binary")
    m_recall = recall_score(y_test, y_predic, average = "binary")
    m_precision = precision_score(y_test, y_predic, average = "binary")
    return m_f1, m_recall, m_precision


def data_splitting(m_df, drop_feature_list):
    # drop non discriminant features
    m_df.drop(drop_feature_list, axis=1, inplace=True)

    # split into normal and anomaly
    df_l1 = m_df[m_df["Label"] == 1]
    df_l0 = m_df[m_df["Label"] == 0]
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
    m_norm_train_df = df_l0[:norm_train_end]  # 60% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]  # 20% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 20% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    m_cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    m_test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    m_norm_train_df = m_norm_train_df.sort_index()
    m_cv_df = m_cv_df.sort_index()
    m_test_df = m_test_df.sort_index()
    gc.collect()

    # save labels and drop labels from data
    m_cv_label = m_cv_df["Label"]
    m_test_label = m_test_df["Label"]
    m_norm_train_df = m_norm_train_df.drop(labels=["Label"], axis=1)
    m_cv_df = m_cv_df.drop(labels=["Label"], axis=1)
    m_test_df = m_test_df.drop(labels=["Label"], axis=1)

    gc.collect()

    return m_norm_train_df, m_cv_df, m_test_df, m_cv_label, m_test_label


def getBestByCV(cv, t_cv_label):

    # prepare data
    m_cv_label = t_cv_label.astype(np.int8)

    # initialize
    m_best_contamination = 0
    m_best_f1 = 0
    m_best_precision = 0
    m_best_recall = 0

    for m_contamination in np.linspace(0.01, 0.1, 15):
        # configure GridSearchCV
        m_ell_model = EllipticEnvelope(contamination = m_contamination)
        m_ell_model.fit(cv, m_cv_label)
        m_pred = m_ell_model.predict(cv)
        m_pred[m_pred == 1] = 0
        m_pred[m_pred == -1] = 1

        m_f1 = f1_score(m_cv_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = precision_score(m_cv_label, m_pred, average="binary")
            m_best_recall = recall_score(m_cv_label, m_pred, average="binary")

    return m_best_contamination, m_best_f1, m_best_precision, m_best_recall


def getBestByNormalCV(t_normal, cv, t_cv_label):

    # prepare data
    m_cv_label = t_cv_label.astype(np.int8)

    # initialize
    m_best_model = EllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_precision = -1
    m_best_recall = -1
    m_pred = []

    for m_contamination in np.linspace(0.01, 0.2, 20):
        m_ell_model = EllipticEnvelope(contamination = m_contamination)
        m_ell_model.fit(t_normal)
        m_pred = m_ell_model.predict(cv)
        m_pred[m_pred == 1] = 0
        m_pred[m_pred == -1] = 1

        m_f1 = f1_score(m_cv_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_model = m_ell_model
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = precision_score(m_cv_label, m_pred, average="binary")
            m_best_recall = recall_score(m_cv_label, m_pred, average="binary")

    m_cm = confusion_matrix(t_cv_label, m_pred)

    return m_best_model, m_best_contamination, m_best_f1, m_best_precision, m_best_recall, m_cm


start_time = time.time()

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
    'Label': 'uint8'}

drop_features = {
    'drop_features00': []
    # 'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes', 'Proto'],
    # 'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes'],
    # 'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'Proto'],
    # 'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
}

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum/')
# raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw_sum/')
raw_directory = os.fsencode(raw_path)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum/')
# pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_fast/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# for each feature set
for features_key, value in drop_features.items():

    # Initialize labels
    ee_test_label = []
    ee_pred_test_label = []

    # for each file/case
    for sample_file in file_list:

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
        norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_features[features_key])

        # Cross-Validation and model selection
        ell_model, best_contamination, best_f1, best_precision, best_recall, best_cm = getBestByNormalCV(norm_train_df, cv_df, cv_label)
        print('###[mcd][', features_key, '] CV. Cont:', best_contamination,', F1:', best_f1, ', Recall:', best_recall, ', Precision:', best_precision, ', TN:', best_cm[0,0], ', FP:', best_cm[0,1], ', FN:', best_cm[1,0], ', TP:', best_cm[1,1])

        # Test
        test_label = test_label.astype(np.int8)
        pred_test_label = ell_model.predict(test_df)
        pred_test_label[pred_test_label == 1] = 0
        pred_test_label[pred_test_label == -1] = 1

        # print results
        f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
        cm = confusion_matrix(test_label, pred_test_label)
        true_unique, true_counts = np.unique(test_label, return_counts=True)
        pred_unique, pred_counts = np.unique(pred_test_label, return_counts=True)
        print('###[mcd][', features_key, '] Test. F1:', f1, ', Recall:', Recall, ', Precision:', Precision, ', TN:', cm[0,0], ', FP:', cm[0,1], ', FN:', cm[1,0], ', TP:', cm[1,1], ', True:', dict(zip(true_unique, true_counts)), ', Pred:', dict(zip(pred_unique, pred_counts)))

        # save results for total evaluation later
        ee_test_label.extend(test_label)
        ee_pred_test_label.extend(pred_test_label)

    f1, Recall, Precision = get_classification_report(ee_test_label, ee_pred_test_label)
    print('###[mcd][', features_key, '] Test Full. F1:',f1,', Recall:',Recall,', Precision:',Precision)
print("--- %s seconds ---" % (time.time() - start_time))