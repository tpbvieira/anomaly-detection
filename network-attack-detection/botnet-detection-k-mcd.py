# coding=utf-8
import os, sys, gc, time, warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, recall_score, precision_score
from k_mcd_outlier_detection import MEllipticEnvelope
warnings.filterwarnings("ignore")


def data_cleasing(m_df):
    # data cleasing, feature engineering and save clean data into pickles

    print('### Data Cleasing and Feature Engineering')
    le = preprocessing.LabelEncoder()

    # [Protocol] - Discard ipv6-icmp and categorize
    m_df = m_df[m_df.Proto != 'ipv6-icmp']
    m_df['Proto'] = m_df['Proto'].fillna('-')
    m_df['Proto'] = le.fit_transform(m_df['Proto'])

    # [Label] - Categorize
    anomalies = m_df.Label.str.contains('Botnet')
    normal = np.invert(anomalies)
    m_df.loc[anomalies, 'Label'] = np.uint8(1)
    m_df.loc[normal, 'Label'] = np.uint8(0)
    m_df['Label'] = pd.to_numeric(m_df['Label'])

    # [Dport] - replace NaN with 0 port number
    m_df['Dport'] = m_df['Dport'].fillna('0')
    m_df['Dport'] = m_df['Dport'].apply(lambda x: int(x, 0))

    # [sport] - replace NaN with 0 port number
    try:
        m_df['Sport'] = m_df['Sport'].fillna('0')
        m_df['Sport'] = m_df['Sport'].str.replace('.*x+.*', '0')
        m_df['Sport'] = m_df['Sport'].apply(lambda x: int(x, 0))
    except:
        print("Unexpected error:", sys.exc_info()[0])

    # [sTos] - replace NaN with "10" and convert to int
    m_df['sTos'] = m_df['sTos'].fillna('10')
    m_df['sTos'] = m_df['sTos'].astype(int)

    # [dTos] - replace NaN with "10" and convert to int
    m_df['dTos'] = m_df['dTos'].fillna('10')
    m_df['dTos'] = m_df['dTos'].astype(int)

    # [State] - replace NaN with "-" and categorize
    m_df['State'] = m_df['State'].fillna('-')
    m_df['State'] = le.fit_transform(m_df['State'])

    # [Dir] - replace NaN with "-" and categorize
    m_df['Dir'] = m_df['Dir'].fillna('-')
    m_df['Dir'] = le.fit_transform(m_df['Dir'])

    # [SrcAddr] Extract subnet features and categorize
    m_df['SrcAddr'] = m_df['SrcAddr'].fillna('0.0.0.0')

    # [DstAddr] Extract subnet features
    m_df['DstAddr'] = m_df['DstAddr'].fillna('0.0.0.0')

    # [StartTime] - Parse to datatime, reindex based on StartTime, but first drop the ns off the time stamps
    m_df['StartTime'] = m_df['StartTime'].apply(lambda x: x[:19])
    m_df['StartTime'] = pd.to_datetime(m_df['StartTime'])

    m_df = m_df.set_index('StartTime')

    gc.collect()

    return m_df


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


def getBestBySemiSupervKurtosisCV(t_normal_df, t_cv_df, t_cv_label):
    m_cv_label = t_cv_label.astype(np.int8)

    # initialize
    m_best_model = MEllipticEnvelope()
    m_best_contamination = -1
    m_best_f1 = -1
    m_best_precision = -1
    m_best_recall = -1

    # configure GridSearchCV
    for m_contamination in np.linspace(0.01, 0.2, 20):
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
    # 'drop_features00': []
    'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes', 'Proto'],
    'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes'],
    'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'Proto'],
    'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
}

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw/')
raw_directory = os.fsencode(raw_path)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/')
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
        ell_model, best_contamination, best_f1, best_precision, best_recall = getBestBySemiSupervKurtosisCV(norm_train_df, cv_df, cv_label)
        print('###[k-mcd][', features_key, '] Cross-Validation. Contamination:', best_contamination,', F1:', best_f1, ', Recall:', best_recall, ', Precision:', best_precision)

        # Test
        pred_test_label = ell_model.kurtosis_prediction(test_df)

        # print results
        f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
        print('###[k-mcd][', features_key, '] Test. F1:', f1,', Recall:', Recall, ', Precision:', Precision)
        # unique, counts = np.unique(test_label, return_counts=True)
        # print(dict(zip(unique, counts)))
        # unique, counts = np.unique(pred_test_label, return_counts=True)
        # print(dict(zip(unique, counts)))

        # save results for total evaluation later
        ee_test_label.extend(test_label)
        ee_pred_test_label.extend(pred_test_label)

    f1, Recall, Precision = get_classification_report(ee_test_label, ee_pred_test_label)
    print('###[k-mcd][', features_key, '] Test Full. F1:', f1, ', Recall:', Recall, ', Precision:', Precision)
print("--- %s seconds ---" % (time.time() - start_time))