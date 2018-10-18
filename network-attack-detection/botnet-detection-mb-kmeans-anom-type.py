# coding=utf-8
'''
minibactch-kmeans implementation for normal vs attack or normal vs CC data
'''
import pandas as pd
import numpy as np
import os
import sys
import gc
import ipaddress
import time
from functools import reduce
from scipy.stats import multivariate_normal
from sklearn import preprocessing, mixture
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.cluster import MiniBatchKMeans


def data_cleasing(df):
        
    print('### Data Cleasing and Feature Engineering')
    le = preprocessing.LabelEncoder()
    
    # dropping ipv6 and icmp
    try:
        # print('dropping ipv6 and icmp')
        df = df[df.Proto != 'ipv6']        
        df = df[df.Proto != 'ipv6-icmp']        
        df = df[df.Proto != 'icmp']
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])        

    try:
        # print('Proto')
        df['Proto'] = df['Proto'].fillna('-')
        df['Proto'] = le.fit_transform(df['Proto'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['Proto'].to_csv('error_proto.csv', index=False)

    try:
        print('Label')
        anomalies = df.Label.str.contains('Botnet')
        normal = np.invert(anomalies);
        df.loc[anomalies, 'Label'] = np.uint8(1)
        df.loc[normal, 'Label'] = np.uint8(0)
        df['Label'] = pd.to_numeric(df['Label'])
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['Label'].to_csv('error_label.csv', index=False)

    try:
        print('Dport')
        df['Dport'] = df['Sport'].str.replace('.*x+.*', '0')
        df['Dport'] = df['Dport'].fillna('0')        
        df['Dport'] = df['Dport'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['Dport'].to_csv('error_dport.csv', index=False)
    
    try:
        print('Sport')
        df['Sport'] = df['Sport'].str.replace('.*x+.*', '0')
        df['Sport'] = df['Sport'].fillna('0')        
        df['Sport'] = df['Sport'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info())
        df['Sport'].to_csv('error_sport.csv', index=False)

    try:
        print('sTos')
        df['sTos'] = df['sTos'].fillna('10')
        df['sTos'] = df['sTos'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['Stos'].to_csv('error_stos.csv', index=False)

    try:
        print('dTos')
        df['dTos'] = df['dTos'].fillna('10')
        df['dTos'] = df['dTos'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['dTos'].to_csv('error_dtos.csv', index=False)

    try:
        print('State')
        df['State'] = df['State'].fillna('-')
        df['State'] = le.fit_transform(df['State'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['State'].to_csv('error_state.csv', index=False)

    try:
        print('Dir')
        df['Dir'] = df['Dir'].fillna('-')
        df['Dir'] = le.fit_transform(df['Dir'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['Dir'].to_csv('error_dir.csv', index=False)

    try:
        print('SrcAddr')
        df['SrcAddr'] = df['SrcAddr'].fillna('0.0.0.0')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['SrcAddr'].to_csv('error_srcaddr.csv', index=False)
    
    try:
        print('DstAddr')
        df['DstAddr'] = df['DstAddr'].fillna('0.0.0.0')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['DstAddr'].to_csv('error_dstaddr.csv', index=False)
    
    try:
        print('StartTime')
        df['StartTime'] = df['StartTime'].apply(lambda x: x[:19])
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df = df.set_index('StartTime')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        df['StartTime'].to_csv('error_starttime.csv', index=False)
    
    return df


def classify_ip(ip):
    '''
    str ip - ip address string to attempt to classify. treat ipv6 addresses as N/A
    '''
    try: 
        ip_addr = ipaddress.ip_address(ip)
        if isinstance(ip_addr, ipaddress.IPv6Address):
            return 'ipv6'
        elif isinstance(ip_addr, ipaddress.IPv4Address):
            # split on .
            octs = ip_addr.exploded.split('.')
            if 0 < int(octs[0]) < 127: return 'A'
            elif 127 < int(octs[0]) < 192: return 'B'
            elif 191 < int(octs[0]) < 224: return 'C'
            else: return 'N/A'
    except ValueError:
        return 'N/A'

            
def avg_duration(x):
    return np.average(x)
            

def n_dports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b>1024 else a),x)
n_dports_gt1024.__name__ = 'n_dports>1024'


def n_dports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b<1024 else a),x)
n_dports_lt1024.__name__ = 'n_dports<1024'


def n_sports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b>1024 else a),x)
n_sports_gt1024.__name__ = 'n_sports>1024'


def n_sports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b<1024 else a),x)
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
        if p == 10: count += 1 # tcp == 10
    return count
            

def n_udp(x):
    count = 0
    for p in x: 
        if p == 11: count += 1 # udp == 11
    return count
            

def n_icmp(x):
    count = 0
    for p in x: 
        if p == 1: count += 1 # icmp == 1
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


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariateGaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return p.pdf(dataset)


def print_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average = "binary")
    Recall = recall_score(y_test, y_predic, average = "binary")
    Precision = precision_score(y_test, y_predic, average = "binary")
    print('\tF1 Score: ',f1,', Recall: ',Recall,', Precision: ,',Precision)


def get_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average = "binary")
    Recall = recall_score(y_test, y_predic, average = "binary")
    Precision = precision_score(y_test, y_predic, average = "binary")
    return f1, Recall,Precision


def data_merge_splitting(normal_df, anom_df):

    # Length and indexes
    norm_len = len(normal_df.index)
    norm_train_end = (norm_len * 60) // 100
    norm_cv_start = norm_train_end + 1
    norm_cv_end = (norm_len * 80) // 100
    norm_test_start = norm_cv_end + 1
    anom_len = len(anom_df.index)
    anom_cv_end = anom_len // 2
    anom_test_start = anom_cv_end + 1

    # split normal data
    norm_train_df = normal_df[:norm_train_end]
    norm_cv_df = normal_df[norm_cv_start:norm_cv_end]
    norm_test_df = normal_df[norm_test_start:norm_len]
    
    # split anomalous data
    anom_cv_df  = anom_df[:anom_cv_end]
    anom_test_df = anom_df[anom_test_start:anom_len]

    # CV and test data from concatenation of normal and anomalous data
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)

    # labels
    cv_label_df = cv_df["Label"]
    test_label_df = test_df["Label"]

    # drop label from data
    norm_train_df = norm_train_df.drop(labels = ["Label"], axis = 1)
    cv_df = cv_df.drop(labels = ["Label"], axis = 1)
    test_df = test_df.drop(labels = ["Label"], axis = 1)
    
    return norm_train_df, cv_df, test_df, cv_label_df, test_label_df


def getBestByCV(X_train, X_cv, labels):
    # select the best epsilon (threshold) and number of clusters
    
    # initialize
    best_epsilon = 0
    best_cluster_size = 0
    best_batch_size = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    for m_clusters in np.arange(1, 10, 2):

        for m_batch_size in range(10, 100, 10): 

            mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=m_clusters, batch_size=m_batch_size, n_init=10, max_no_improvement=10).fit(X_train)
            
            X_cv_clusters = mbkmeans.predict(X_cv)
            X_cv_clusters_centers = mbkmeans.cluster_centers_

            dist = [np.linalg.norm(x-y) for x,y in zip(X_cv.as_matrix(), X_cv_clusters_centers[X_cv_clusters])]

            y_pred = np.array(dist)        

            for m_epsilon in np.arange(70, 95, 2):
                y_pred[dist >= np.percentile(dist,m_epsilon)] = 1
                y_pred[dist < np.percentile(dist,m_epsilon)] = 0
            
                f1 = f1_score(labels, y_pred, average = "binary")
                Recall = recall_score(labels, y_pred, average = "binary")
                Precision = precision_score(labels, y_pred, average = "binary") 

                if f1 > best_f1:
                    best_cluster_size = m_clusters
                    best_batch_size = m_batch_size
                    best_epsilon = m_epsilon
                    best_f1 = f1
                    best_precision = Precision
                    best_recall = Recall

    return best_cluster_size, best_batch_size, best_epsilon, best_f1, best_precision, best_recall


# track execution time
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
            'Label': 'str'}

# feature selection
drop_features = {
    'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes', 'Proto'],
    'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes'],
    'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'Proto'],
    'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
}

raw_normal_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_normal/')
raw_normal_directory = os.fsencode(raw_normal_path)
raw_normal_files = os.listdir(raw_normal_directory)
raw_anom_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_cc/')
raw_anom_directory = os.fsencode(raw_anom_path)
raw_anom_files = os.listdir(raw_anom_directory)

pkl_normal_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/pkl_normal/')
pkl_normal_directory = os.fsencode(pkl_normal_path)
pkl_anom_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/pkl_cc/')
pkl_anom_directory = os.fsencode(pkl_anom_path)

# for each feature set
for features_key, value in drop_features.items():

    all_normal_df = None
    all_anom_df = None

    for sample_file in raw_normal_files:
        
        raw_normal_file_path = os.path.join(raw_normal_directory, sample_file).decode('utf-8')    
        raw_anom_file_path = os.path.join(raw_anom_directory, sample_file).decode('utf-8')

        pkl_normal_file_path = os.path.join(pkl_normal_directory, sample_file).decode('utf-8')
        pkl_anom_file_path = os.path.join(pkl_anom_directory, sample_file).decode('utf-8')        

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_normal_file_path):
            print("## PKL Normal File: ", pkl_normal_file_path)
            normal_df = pd.read_pickle(pkl_normal_file_path)
        else:
            print("## Raw Normal File: ", raw_normal_file_path)
            normal_df = pd.read_csv(raw_normal_file_path, low_memory=True, header=0, dtype=column_types)
            normal_df = data_cleasing(normal_df)
            normal_df.to_pickle(pkl_normal_file_path)            
        gc.collect()

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_anom_file_path):
            print("## PKL Anomalous File: ", pkl_anom_file_path)
            anom_df = pd.read_pickle(pkl_anom_file_path)
        else:
            print("## Raw Anomalous File: ", raw_anom_file_path)            
            anom_df = pd.read_csv(raw_anom_file_path, low_memory=True, header=0, dtype=column_types)
            anom_df = data_cleasing(anom_df)
            anom_df.to_pickle(pkl_anom_file_path)
        gc.collect()

        if all_normal_df is None:
            all_normal_df = normal_df
        else:
            all_normal_df.append(normal_df)

        if all_anom_df is None:
            all_anom_df = anom_df
        else:
            all_anom_df.append(anom_df)

    # drop features
    all_normal_df.drop(drop_features[features_key], axis=1, inplace=True)
    all_anom_df.drop(drop_features[features_key], axis=1, inplace=True)

    # data merge and splitting
    train_df, cv_df, test_df, cv_label_df, test_label_df = data_merge_splitting(all_normal_df, all_anom_df)

    # Cross-Validation
    b_clusters, b_batch, b_epsilon, b_f1, b_precision, b_recall = getBestByCV(train_df, cv_df, cv_label_df)
    print('###[MB-KMeans][', features_key, '] Cross-Validation. Clusters:', b_clusters, ', Batch:', b_batch, ', Epslilon:', b_epsilon, ',F1:', b_f1, ', Recall:', b_recall, ', Precision:',
          b_precision)

    # Training - estimate clusters (anomalous or normal) for training    
    mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=b_clusters, batch_size=b_batch, n_init=10, max_no_improvement=10).fit(train_df)

    # Test prediction
    test_clusters = mbkmeans.predict(test_df)
    test_clusters_centers = mbkmeans.cluster_centers_
    dist = [np.linalg.norm(x-y) for x,y in zip(test_df.as_matrix(), test_clusters_centers[test_clusters])]
    pred_test_label = np.array(dist)
    pred_test_label[dist >= np.percentile(dist, b_epsilon)] = 1
    pred_test_label[dist < np.percentile(dist, b_epsilon)] = 0
    test_label = test_label_df.astype(int).values

    # print results
    f1, Recall, Precision = get_classification_report(test_label_df.astype(int).values, pred_test_label)
    print('###[MB-KMeans][', features_key, '] Test. F1:', f1, ', Recall:', Recall, ', Precision:', Precision)
print("--- %s seconds ---" % (time.time() - start_time))