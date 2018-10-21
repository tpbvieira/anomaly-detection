# coding=utf-8
import os, sys, gc, ipaddress, time, warnings
import pandas as pd
import numpy as np
from functools import reduce
from sklearn import preprocessing
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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


def classify_ip(ip):
    """
    str ip - ip address string to attempt to classify. treat ipv6 addresses as N/A
    """
    try:
        ip_addr = ipaddress.ip_address(ip)
        if isinstance(ip_addr, ipaddress.IPv6Address):
            return 'ipv6'
        elif isinstance(ip_addr, ipaddress.IPv4Address):
            # split on .
            octs = ip_addr.exploded.split('.')
            if 0 < int(octs[0]) < 127:
                return 'A'
            elif 127 < int(octs[0]) < 192:
                return 'B'
            elif 191 < int(octs[0]) < 224:
                return 'C'
            else:
                return 'N/A'
    except ValueError:
        return 'N/A'


def avg_duration(x):
    return np.average(x)


def n_dports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a, b: a + b if b > 1024 else a), x)


n_dports_gt1024.__name__ = 'n_dports>1024'


def n_dports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a, b: a + b if b < 1024 else a), x)


n_dports_lt1024.__name__ = 'n_dports<1024'


def n_sports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a, b: a + b if b > 1024 else a), x)


n_sports_gt1024.__name__ = 'n_sports>1024'


def n_sports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a, b: a + b if b < 1024 else a), x)


n_sports_lt1024.__name__ = 'n_sports<1024'


def label_atk_v_norm(x):
    for l in x:
        if l == 1: return 1
    return 0


label_atk_v_norm.__name__ = 'label'


def background_flow_count(x):
    count = 0
    for l in x:
        if l == 0: count += 1
    return count


def normal_flow_count(x):
    if x.size == 0: return 0
    count = 0
    for l in x:
        if l == 0: count += 1
    return count


def n_conn(x):
    return x.size


def n_tcp(x):
    count = 0
    for p in x:
        if p == 10: count += 1  # tcp == 10
    return count


def n_udp(x):
    count = 0
    for p in x:
        if p == 11: count += 1  # udp == 11
    return count


def n_icmp(x):
    count = 0
    for p in x:
        if p == 1: count += 1  # icmp == 1
    return count


def n_s_a_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'A': count += 1
    return count


def n_d_a_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'A': count += 1
    return count


def n_s_b_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'B': count += 1
    return count


def n_d_b_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'A': count += 1
    return count


def n_s_c_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'C': count += 1
    return count


def n_d_c_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'C': count += 1
    return count


def n_s_na_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'N/A': count += 1
    return count


def n_d_na_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'N/A': count += 1
    return count


def n_ipv6(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'ipv6': count += 1
    return count


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
    m_cv_label[m_cv_label == 1] = -1
    m_cv_label[m_cv_label == 0] = 1

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

        m_f1 = f1_score(m_cv_label, m_pred, average="binary")
        m_recall = recall_score(m_cv_label, m_pred, average="binary")
        m_precision = precision_score(m_cv_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = m_precision
            m_best_recall = m_recall

    return m_best_contamination, m_best_f1, m_best_precision, m_best_recall


def getBestByNormalCV(t_normal, cv, t_cv_label):

    # prepare data
    m_cv_label = t_cv_label.astype(np.int8)
    m_cv_label[m_cv_label == 1] = -1
    m_cv_label[m_cv_label == 0] = 1

    # initialize
    m_best_contamination = 0
    m_best_f1 = 0
    m_best_precision = 0
    m_best_recall = 0

    for m_contamination in np.linspace(0.01, 0.1, 15):
        # configure GridSearchCV
        m_ell_model = EllipticEnvelope(contamination = m_contamination)
        m_ell_model.fit(t_normal)
        m_pred = m_ell_model.predict(cv)

        m_f1 = f1_score(m_cv_label, m_pred, average="binary")
        m_recall = recall_score(m_cv_label, m_pred, average="binary")
        m_precision = precision_score(m_cv_label, m_pred, average="binary")

        if m_f1 > m_best_f1:
            m_best_contamination = m_contamination
            m_best_f1 = m_f1
            m_best_precision = m_precision
            m_best_recall = m_recall

    return m_best_contamination, m_best_f1, m_best_precision, m_best_recall


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

# feature selection
drop_features = {
    'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes', 'Proto'],
    'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes'],
    'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'Proto'],
    'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
}

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/raw/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl/')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_features.items():

    # Initialize labels
    ee_test_label = []
    ee_pred_test_label = []

    for sample_file in raw_files:

        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')

        # read pickle file with pandas or...
        if os.path.isfile(pkl_file_path):
            print("## Sample File: ", pkl_file_path)
            df = pd.read_pickle(pkl_file_path)
        else:  # load raw file and save clean data into pickles
            print("## Sample File: ", raw_file_path)
            raw_df = pd.read_csv(raw_file_path, header=0, dtype=column_types)
            df = data_cleasing(raw_df)
            df.to_pickle(pkl_file_path)
        gc.collect()

        # data splitting
        norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_features[features_key])

        # Cross-Validation
        best_contamination, best_f1, best_precision, best_recall = getBestByNormalCV(norm_train_df, cv_df, cv_label)
        print('###[EllipticEnvelope][', features_key, '] Cross-Validation. Contamination:',best_contamination,',F1:', best_f1, ', Recall:', best_recall, ', Precision:', best_precision)

        # Test
        test_label = test_label.astype(np.int8)
        test_label[test_label == 1] = -1
        test_label[test_label == 0] = 1
        ell_model = EllipticEnvelope(contamination=best_contamination)
        ell_model.fit(test_df, test_label)
        pred_test_label = ell_model.predict(test_df)

        # print results
        f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
        print('###[EllipticEnvelope][', features_key, '] Test. F1:', f1, ', Recall:', Recall, ', Precision:', Precision)

        # save results for total evaluation later
        ee_test_label.extend(test_label)
        ee_pred_test_label.extend(pred_test_label)

    f1, Recall, Precision = get_classification_report(ee_test_label, ee_pred_test_label)
    print('###[EllipticEnvelope][', features_key, '] Test Full. F1:',f1,', Recall:',Recall,', Precision:',Precision)
print("--- %s seconds ---" % (time.time() - start_time))