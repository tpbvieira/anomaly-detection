# coding=utf-8
import gc
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score, recall_score, precision_score
warnings.filterwarnings("ignore")


# features
ctu13_raw_column_types = {
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

ctu13_drop_raw_features = {
    'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'Proto', 'TotBytes', 'SrcBytes'],
    'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'TotBytes', 'SrcBytes'],
    'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'Proto', 'SrcBytes'],
    'drop_features05': ['SrcAddr', 'DstAddr', 'sTos', 'Proto'],
    'drop_features04': ['SrcAddr', 'DstAddr']
}

ctu13_drop_agg_features = {
    'drop_features00': []
}


def ctu13_data_cleasing(m_df):

    # data cleasing, feature engineering and save clean data into pickles
    print('### Data Cleasing and Feature Engineering')
    label_encoder = preprocessing.LabelEncoder()

    try:
        print('Proto')
        m_df['Proto'] = m_df['Proto'].fillna('-')
        m_df['Proto'] = label_encoder.fit_transform(m_df['Proto'])
        # le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['Proto'].head())
        m_df['Proto'].to_csv('error_proto.csv', index=False)

    try:
        print('Label')
        anomalies = m_df.Label.str.contains('Botnet')
        normal = np.invert(anomalies)
        m_df.loc[anomalies, 'Label'] = np.uint8(1)
        m_df.loc[normal, 'Label'] = np.uint8(0)
        m_df['Label'] = pd.to_numeric(m_df['Label'])
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['Label'].head())
        m_df['Label'].to_csv('error_label.csv', index=False)

    try:
        print('Dport')
        m_df['Dport'] = m_df['Sport'].str.replace('.*x+.*', '0')
        m_df['Dport'] = m_df['Dport'].fillna('0')
        m_df['Dport'] = m_df['Dport'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['Dport'].head())
        m_df['Dport'].to_csv('error_dport.csv', index=False)

    try:
        print('Sport')
        m_df['Sport'] = m_df['Sport'].str.replace('.*x+.*', '0')
        m_df['Sport'] = m_df['Sport'].fillna('0')
        m_df['Sport'] = m_df['Sport'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info())
        print(m_df.head())
        print(m_df['Sport'].head())
        m_df['Sport'].to_csv('error_sport.csv', index=False)

    try:
        print('sTos')
        m_df['sTos'] = m_df['sTos'].fillna('10')
        m_df['sTos'] = m_df['sTos'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['Stos'].head())
        m_df['Stos'].to_csv('error_stos.csv', index=False)

    try:
        print('dTos')
        m_df['dTos'] = m_df['dTos'].fillna('10')
        m_df['dTos'] = m_df['dTos'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['dTos'].head())
        m_df['dTos'].to_csv('error_dtos.csv', index=False)

    try:
        print('State')
        m_df['State'] = m_df['State'].fillna('-')
        m_df['State'] = label_encoder.fit_transform(m_df['State'])
        # le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['State'].head())
        m_df['State'].to_csv('error_state.csv', index=False)

    try:
        print('Dir')
        m_df['Dir'] = m_df['Dir'].fillna('-')
        m_df['Dir'] = label_encoder.fit_transform(m_df['Dir'])
        # le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['Dir'].head())
        m_df['Dir'].to_csv('error_dir.csv', index=False)

    try:
        print('SrcAddr')
        m_df['SrcAddr'] = m_df['SrcAddr'].fillna('0.0.0.0')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['SrcAddr'].head())
        m_df['SrcAddr'].to_csv('error_srcaddr.csv', index=False)

    try:
        print('DstAddr')
        m_df['DstAddr'] = m_df['DstAddr'].fillna('0.0.0.0')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['DstAddr'].head())
        m_df['DstAddr'].to_csv('error_dstaddr.csv', index=False)

    try:
        print('StartTime')
        m_df['StartTime'] = m_df['StartTime'].apply(lambda x: x[:19])
        m_df['StartTime'] = pd.to_datetime(m_df['StartTime'])
        m_df = m_df.set_index('StartTime')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['StartTime'].head())
        m_df['StartTime'].to_csv('error_starttime.csv', index=False)

    gc.collect()
    return m_df


def data_splitting_20_40_40(df, drop_features):
    # Data splitting

    # drop non discriminant features
    df.drop(drop_features, axis=1, inplace=True)

    df['PktsRate'] = df.TotPkts / df.Dur
    df['BytesRate'] = df.TotBytes / df.Dur
    df['MeanPktsRate'] = df.TotBytes / df.TotPkts

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)                           # total number of anomalous flows
    anom_train_end = anom_len // 2                  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1              # 50% of anomalous for testing
    norm_len = len(df_l0)                           # total number of normal flows
    norm_train_end = (norm_len * 20) // 100         # 20% of normal for training
    norm_cv_start = norm_train_end + 1              # 40% of normal for cross validation
    norm_cv_end = (norm_len * 60) // 100            # 40% of normal for cross validation
    norm_test_start = norm_cv_end + 1               # 40% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]             # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]    # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]          # 20% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]   # 40% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 40% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    norm_train_df = norm_train_df.sort_index()
    cv_df = cv_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop label from data
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]
    norm_train_df = norm_train_df.drop(labels=["Label"], axis=1)
    cv_df = cv_df.drop(labels=["Label"], axis=1)
    test_df = test_df.drop(labels=["Label"], axis=1)

    gc.collect()

    return norm_train_df, cv_df, test_df, cv_label, test_label


def data_splitting_34_33_33(df, m_col_list):

    if 'PktsRate' in m_col_list:
        df['PktsRate'] = df.TotPkts / df.Dur
        df['PktsRate'] = df['PktsRate'].replace(-np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].replace(np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].fillna(-1)

    if 'BytesRate' in m_col_list:
        df['BytesRate'] = df.TotBytes / df.Dur
        df['BytesRate'] = df['BytesRate'].replace(-np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].replace(np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].fillna(-1)

    if 'MeanPktsRate' in m_col_list:
        df['MeanPktsRate'] = df.TotBytes / df.TotPkts
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(-np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].fillna(-1)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)                           # total number of anomalous flows
    anom_train_end = anom_len // 2                  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1              # 50% of anomalous for testing
    norm_len = len(df_l0)                           # total number of normal flows
    norm_train_end = (norm_len * 34) // 100         # 34% of normal for training
    norm_cv_start = norm_train_end + 1              # 33% of normal for cross validation
    norm_cv_end = (norm_len * 67) // 100            # 33% of normal for cross validation
    norm_test_start = norm_cv_end + 1               # 33% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]             # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]    # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]          # 20% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]   # 40% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 40% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    norm_train_df = norm_train_df.sort_index()
    cv_df = cv_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop label from data
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]

    norm_train_df = norm_train_df[m_col_list]
    cv_df = cv_df[m_col_list]
    test_df = test_df[m_col_list]
    gc.collect()

    return norm_train_df, cv_df, test_df, cv_label, test_label


def data_splitting_40_30_30(df, m_col_list):

    if 'PktsRate' in m_col_list:
        df['PktsRate'] = df.TotPkts / df.Dur
        df['PktsRate'] = df['PktsRate'].replace(-np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].replace(np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].fillna(-1)

    if 'BytesRate' in m_col_list:
        df['BytesRate'] = df.TotBytes / df.Dur
        df['BytesRate'] = df['BytesRate'].replace(-np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].replace(np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].fillna(-1)

    if 'MeanPktsRate' in m_col_list:
        df['MeanPktsRate'] = df.TotBytes / df.TotPkts
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(-np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].fillna(-1)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)                           # total number of anomalous flows
    anom_train_end = anom_len // 2                  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1              # 50% of anomalous for testing
    norm_len = len(df_l0)                           # total number of normal flows
    norm_train_end = (norm_len * 40) // 100         # 40% of normal for training
    norm_cv_start = norm_train_end + 1              # 30% of normal for cross validation
    norm_cv_end = (norm_len * 70) // 100            # 30% of normal for cross validation
    norm_test_start = norm_cv_end + 1               # 30% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]             # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]    # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]          # 40% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]   # 30% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 30% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    norm_train_df = norm_train_df.sort_index()
    cv_df = cv_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop label from data
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]

    norm_train_df = norm_train_df[m_col_list]
    cv_df = cv_df[m_col_list]
    test_df = test_df[m_col_list]
    gc.collect()

    return norm_train_df, cv_df, test_df, cv_label, test_label


def data_splitting_60_20_20(df, m_col_list):

    if 'PktsRate' in m_col_list:
        df['PktsRate'] = df.TotPkts / df.Dur
        df['PktsRate'] = df['PktsRate'].replace(-np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].replace(np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].fillna(-1)

    if 'BytesRate' in m_col_list:
        df['BytesRate'] = df.TotBytes / df.Dur
        df['BytesRate'] = df['BytesRate'].replace(-np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].replace(np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].fillna(-1)

    if 'MeanPktsRate' in m_col_list:
        df['MeanPktsRate'] = df.TotBytes / df.TotPkts
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(-np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].fillna(-1)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)                           # total number of anomalous flows
    anom_train_end = anom_len // 2                  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1              # 50% of anomalous for testing
    norm_len = len(df_l0)                           # total number of normal flows
    norm_train_end = (norm_len * 60) // 100         # 60% of normal for training
    norm_cv_start = norm_train_end + 1              # 20% of normal for cross validation
    norm_cv_end = (norm_len * 80) // 100            # 20% of normal for cross validation
    norm_test_start = norm_cv_end + 1               # 20% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]             # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]    # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]          # 60% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]   # 20% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 20% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    norm_train_df = norm_train_df.sort_index()
    cv_df = cv_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop label from data
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]

    norm_train_df = norm_train_df[m_col_list]
    cv_df = cv_df[m_col_list]
    test_df = test_df[m_col_list]
    gc.collect()

    return norm_train_df, cv_df, test_df, cv_label, test_label


def data_splitting_80_10_10(df, m_col_list):

    if 'PktsRate' in m_col_list:
        df['PktsRate'] = df.TotPkts / df.Dur
        df['PktsRate'] = df['PktsRate'].replace(-np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].replace(np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].fillna(-1)

    if 'BytesRate' in m_col_list:
        df['BytesRate'] = df.TotBytes / df.Dur
        df['BytesRate'] = df['BytesRate'].replace(-np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].replace(np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].fillna(-1)

    if 'MeanPktsRate' in m_col_list:
        df['MeanPktsRate'] = df.TotBytes / df.TotPkts
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(-np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].fillna(-1)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)                           # total number of anomalous flows
    anom_train_end = anom_len // 2                  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1              # 50% of anomalous for testing
    norm_len = len(df_l0)                           # total number of normal flows
    norm_train_end = (norm_len * 80) // 100         # 80% of normal for training
    norm_cv_start = norm_train_end + 1              # 10% of normal for cross validation
    norm_cv_end = (norm_len * 90) // 100            # 10% of normal for cross validation
    norm_test_start = norm_cv_end + 1               # 10% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]             # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]    # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]          # 80% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]   # 10% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 10% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    norm_train_df = norm_train_df.sort_index()
    cv_df = cv_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop label from data
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]

    norm_train_df = norm_train_df[m_col_list]
    cv_df = cv_df[m_col_list]
    test_df = test_df[m_col_list]
    gc.collect()

    return norm_train_df, cv_df, test_df, cv_label, test_label


def data_splitting_50_50(df, m_col_list):

    if 'PktsRate' in m_col_list:
        df['PktsRate'] = df.TotPkts / df.Dur
        df['PktsRate'] = df['PktsRate'].replace(-np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].replace(np.inf, np.nan)
        df['PktsRate'] = df['PktsRate'].fillna(-1)

    if 'BytesRate' in m_col_list:
        df['BytesRate'] = df.TotBytes / df.Dur
        df['BytesRate'] = df['BytesRate'].replace(-np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].replace(np.inf, np.nan)
        df['BytesRate'] = df['BytesRate'].fillna(-1)

    if 'MeanPktsRate' in m_col_list:
        df['MeanPktsRate'] = df.TotBytes / df.TotPkts
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(-np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].replace(np.inf, np.nan)
        df['MeanPktsRate'] = df['MeanPktsRate'].fillna(-1)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)                           # total number of anomalous flows
    anom_train_end = anom_len // 2                  # 50% of anomalous for training
    anom_test_start = anom_train_end + 1            # 50% of anomalous for testing
    norm_len = len(df_l0)                           # total number of normal flows
    norm_train_end = norm_len // 2                  # 50% of normal for training
    norm_test_start = norm_train_end + 1            # 50% of normal for testing

    # anomalies split data
    anom_train_df = df_l1[:anom_train_end]
    anom_test_df = df_l1[anom_test_start:anom_len]
    gc.collect()

    # training with normal and anomalous data
    norm_train_df = df_l0[:norm_train_end]
    norm_test_df = df_l0[norm_test_start:norm_len]
    gc.collect()

    # testing with normal and anomalous data
    train_df = pd.concat([norm_train_df, anom_train_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop label from data
    train_label_df = train_df["Label"]
    test_label_df = test_df["Label"]

    train_df = train_df[m_col_list]
    test_df = test_df[m_col_list]
    gc.collect()

    return train_df, train_label_df, test_df, test_label_df


def print_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average="binary")
    m_rc = recall_score(y_test, y_predic, average="binary")
    m_pr = precision_score(y_test, y_predic, average="binary")
    print('\tF1 Score: ', f1, ', Recall: ', m_rc, ', Precision: ,', m_pr)


def get_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average="binary")
    m_rc = recall_score(y_test, y_predic, average="binary")
    m_pr = precision_score(y_test, y_predic, average="binary")
    return f1, m_rc, m_pr
