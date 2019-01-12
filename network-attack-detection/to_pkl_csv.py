# coding=utf-8
import os, gc, time, warnings, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from k_mcd_outlier_detection import MEllipticEnvelope
from utils import get_feature_labels, get_feature_order
warnings.filterwarnings("ignore")


def print_classification_report(y_test, y_predic):
    m_f1 = f1_score(y_test, y_predic, average="binary")
    m_recall = recall_score(y_test, y_predic, average="binary")
    m_precision = precision_score(y_test, y_predic, average="binary")
    print('\tF1 Score: ', m_f1, ', Recall: ', m_recall, ', Precision: ,', m_precision)


def get_classification_report(y_test, y_predic):
    m_f1 = f1_score(y_test, y_predic, average="binary")
    m_recall = recall_score(y_test, y_predic, average="binary")
    m_precision = precision_score(y_test, y_predic, average="binary")
    return m_f1, m_recall, m_precision


def data_splitting(m_df, drop_feature):
    # drop non discriminant features
    m_df.drop(drop_feature, axis=1, inplace=True)

    # split into normal and anomaly
    df_l1 = m_df[m_df["Label"] == 1]
    df_l0 = m_df[m_df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)  # total number of anomalous flows
    anom_train_end = anom_len // 2  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1  # 50% of anomalous for testing
    normal_len = len(df_l0)  # total number of normal flows
    normal_train_end = (normal_len * 60) // 100  # 60% of normal for training
    normal_cv_start = normal_train_end + 1  # 20% of normal for cross validation
    normal_cv_end = (normal_len * 80) // 100  # 20% of normal for cross validation
    normal_test_start = normal_cv_end + 1  # 20% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]  # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]  # 50% of anomalies
    gc.collect()

    # normal split data
    m_normal_train_df = df_l0[:normal_train_end]  # 60% of normal
    normal_cv_df = df_l0[normal_cv_start:normal_cv_end]  # 20% of normal
    normal_test_df = df_l0[normal_test_start:normal_len]  # 20% of normal
    gc.collect()

    # CV and test data. train data is normal_train_df
    m_cv_df = pd.concat([normal_cv_df, anom_cv_df], axis=0)
    m_test_df = pd.concat([normal_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    m_normal_train_df = m_normal_train_df.sort_index()
    m_cv_df = m_cv_df.sort_index()
    m_test_df = m_test_df.sort_index()
    gc.collect()

    # save labels and drop labels from data
    m_cv_label = m_cv_df["Label"]
    m_test_label = m_test_df["Label"]
    m_normal_train_df = m_normal_train_df.drop(labels=["Label"], axis=1)
    m_cv_df = m_cv_df.drop(labels=["Label"], axis=1)
    m_test_df = m_test_df.drop(labels=["Label"], axis=1)

    gc.collect()

    return m_normal_train_df, m_cv_df, m_test_df, m_cv_label, m_test_label


def getBestBySemiSupervCV(t_normal_df, t_cv_df, t_cv_label):
    m_cv_label = t_cv_label.astype(np.int8)

    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_precision = -1
    m_best_recall = -1

    # configure GridSearchCV
    for m_contamination in np.linspace(0.01, 0.2, 2):
        m_ell_model = MEllipticEnvelope(contamination=m_contamination)
        m_ell_model.fit(t_normal_df)
        m_pred = m_ell_model.predict(t_cv_df)
        m_pred[m_pred == 1] = 0
        m_pred[m_pred == -1] = 1

        m_f1 = f1_score(m_cv_label, m_pred, average="binary")
        m_recall = recall_score(m_cv_label, m_pred, average="binary")
        m_precision = precision_score(m_cv_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_model = m_ell_model
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = m_precision
            m_best_recall = m_recall
            # print('###[EllipticEnvelope] Cross-Validation. Contamination:', m_contamination, ',F1:', m_f1, ', Recall:', m_recall, ', Precision:', m_precision)

    return m_best_model, m_best_contamination, m_best_f1, m_best_precision, m_best_recall


def getBestBySemiSupervKurtosisCV(t_normal_df, t_cv_df, t_cv_label):
    m_cv_label = t_cv_label.astype(np.int8)

    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_precision = -1
    m_best_recall = -1

    # configure GridSearchCV
    for m_contamination in np.linspace(0.01, 0.2, 2):
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
            # print('###[EllipticEnvelope] Cross-Validation. Contamination:', m_contamination, ',F1:', m_f1, ', Recall:', m_recall, ', Precision:', m_precision)

    return m_best_model, m_best_contamination, m_best_f1, m_best_precision, m_best_recall


def getBestBySupervCV(t_normal_df, t_cv_df, t_cv_label):
    m_normal_train_df = t_normal_df.copy()
    m_cv_df = t_cv_df.copy()

    m_normal_train_df['Label'] = 0
    m_cv_df['Label'] = t_cv_label.astype(np.int8)
    m_train_df = m_normal_train_df.append(m_cv_df)
    m_train_df = m_train_df.sort_index()
    gc.collect()

    # Length and indexes
    m_total_len = len(m_train_df)
    m_train_end = (m_total_len * 80) // 100
    m_test_start = m_train_end + 1

    m_train = m_train_df[:m_train_end]
    m_test = m_train_df[m_test_start:]

    m_test_label = m_test['Label']
    m_train = m_train.drop(labels=["Label"], axis=1)
    m_test = m_test.drop(labels=["Label"], axis=1)

    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_precision = -1
    m_best_recall = -1

    # configure GridSearchCV
    for m_contamination in np.linspace(0.01, 0.2, 2):
        m_ell_model = MEllipticEnvelope(contamination=m_contamination)
        m_ell_model.fit(m_train)
        m_pred = m_ell_model.predict(m_test)
        m_pred[m_pred == 1] = 0
        m_pred[m_pred == -1] = 1

        m_f1 = f1_score(m_test_label, m_pred, average="binary")
        m_recall = recall_score(m_test_label, m_pred, average="binary")
        m_precision = precision_score(m_test_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_model = m_ell_model
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = m_precision
            m_best_recall = m_recall
            # print('###[EllipticEnvelope] Cross-Validation. Contamination:', m_contamination, ',F1:', m_f1, ', Recall:', m_recall, ', Precision:', m_precision)
    return m_best_model, m_best_contamination, m_best_f1, m_best_precision, m_best_recall


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

pkl_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/network-attack-detection/BinetflowTrainer-master/saved_data')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# for each file/case
for sample_file in file_list:
    print(sample_file)
    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')

    if os.path.isfile(pkl_file_path):
        print("## Sample File: ", pkl_file_path)
        with open(pkl_file_path, 'rb') as f:
            summaries = pickle.load(f)
            feature, label = get_feature_labels(summaries)
            df_data = pd.DataFrame(data=feature, columns=get_feature_order())
            df_label = pd.DataFrame(data=label, columns=['Label'])
            df = pd.concat([df_data, df_label], axis=1)

            new_pkl_file_path = os.path.join(pkl_directory, sample_file[6:]).decode('utf-8')
            df.to_pickle(new_pkl_file_path)

            new_csv_file_path = os.path.splitext(new_pkl_file_path)[0] + '.csv'
            df.to_csv(new_csv_file_path, sep=',')

            # # data splitting
            # norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, [])
            #
            # # Cross-Validation and model selection
            # ell_model, best_contamination, best_f1, best_precision, best_recall = getBestBySemiSupervKurtosisCV(norm_train_df, cv_df, cv_label)
            # print('###[Kurtosis-MCD] Cross-Validation. Contamination:', best_contamination,', F1:', best_f1, ', Recall:', best_recall, ', Precision:', best_precision)
            #
            # # Test
            # pred_test_label = ell_model.kurtosis_prediction(test_df)
            #
            # # print results
            # f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
            # print('###[Kurtosis-MCD] Test. F1:', f1,', Recall:', Recall, ', Precision:', Precision)
    gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))