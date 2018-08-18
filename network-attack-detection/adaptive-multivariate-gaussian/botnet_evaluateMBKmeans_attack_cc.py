import glob
import pandas as pd
import numpy as np
import os
import sys
import gc
import datetime as dt
import seaborn as sns
import matplotlib.gridspec as gridspec
import ipaddress
import random as rnd
import plotly.graph_objs as go
import lime
import lime.lime_tabular
import itertools
from pandas.tools.plotting import scatter_matrix
from functools import reduce
from numpy import genfromtxt
from scipy import linalg
from scipy.stats import multivariate_normal
from sklearn import preprocessing, mixture
from sklearn.metrics import classification_report, average_precision_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


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
    # print('### Classification report:')
    # print(classification_report(y_test, y_predic))

    # print('\tAverage Precision = ' + str(average_precision_score(y_test, y_predic)))

    # print('\n### Binary F1 Score, Recall and Precision:')
    f = f1_score(y_test, y_predic, average = "binary")
    Recall = recall_score(y_test, y_predic, average = "binary")
    Precision = precision_score(y_test, y_predic, average = "binary")
    print('\tF1 Score %f' %f)
    # print('\tRecall Score %f' %Recall)
    # print('\tPrecision Score %f' %Precision)

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

def data_splitting(normal_df, anom_df):

    # Length and indexes
    norm_len = len(normal_df.index)
    anom_len = len(anom_df.index)
    anom_train_end = anom_len // 2
    anom_cv_start = anom_train_end + 1
    norm_train_end = (norm_len * 60) // 100
    norm_cv_start = norm_train_end + 1
    norm_cv_end = (norm_len * 80) // 100
    norm_test_start = norm_cv_end + 1

    # anomalies split data
    anom_cv_df  = anom_df[:anom_train_end]
    anom_test_df = anom_df[anom_cv_start:anom_len]

    # normal split data
    norm_train_df = normal_df[:norm_train_end]
    norm_cv_df = normal_df[norm_cv_start:norm_cv_end]
    norm_test_df = normal_df[norm_test_start:norm_len]

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)

    # labels
    cv_label_df = cv_df["Label"]
    test_label_df = test_df["Label"]

    # drop label
    norm_train_df = norm_train_df.drop(labels = ["Label"], axis = 1)
    cv_df = cv_df.drop(labels = ["Label"], axis = 1)
    test_df = test_df.drop(labels = ["Label"], axis = 1)
    
    return norm_train_df, cv_df, test_df, cv_label_df, test_label_df


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
    'drop_features01':['SrcAddr','DstAddr','sTos','Sport','Proto','TotBytes','SrcBytes'],
    'drop_features02':['SrcAddr','DstAddr','sTos','Sport','TotBytes','SrcBytes'],
    'drop_features03':['SrcAddr','DstAddr','sTos','Sport','Proto','SrcBytes'],
    'drop_features04':['SrcAddr','DstAddr','sTos','Proto']
}

raw_normal_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_normal/')
raw_normal_directory = os.fsencode(raw_normal_path)
raw_normal_files = os.listdir(raw_normal_directory)
# print("## Directory: ", raw_normal_directory)
# print("## Files: ", raw_normal_files)
raw_anom_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_cc/')
raw_anom_directory = os.fsencode(raw_anom_path)
raw_anom_files = os.listdir(raw_anom_directory)

# pickle files have the same names
pkl_normal_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/pkl_normal/')
pkl_normal_directory = os.fsencode(pkl_normal_path)
pkl_anom_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/pkl_cc/')
pkl_anom_directory = os.fsencode(pkl_anom_path)

# for each feature set
for features_key, value in drop_features.items():

    # Initialize labels
    mbkmeans_test_label = []
    mbkmeans_pred_test_label = []

    for sample_file in raw_normal_files:
        pkl_normal_file_path = os.path.join(pkl_normal_directory, sample_file).decode('utf-8')
        pkl_anom_file_path = os.path.join(pkl_anom_directory, sample_file).decode('utf-8')
        raw_normal_file_path = os.path.join(raw_normal_directory, sample_file).decode('utf-8')    
        raw_anom_file_path = os.path.join(raw_anom_directory, sample_file).decode('utf-8')

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_normal_file_path):
            print("## PKL Normal File: ", pkl_normal_file_path)
            normal_df = pd.read_pickle(pkl_normal_file_path)
        else:
            print("## Normal File: ", raw_normal_file_path)            
            normal_df = pd.read_csv(raw_normal_file_path, low_memory=True, header = 0, dtype=column_types)
            normal_df = data_cleasing(normal_df)
            normal_df.to_pickle(pkl_normal_file_path)            
        gc.collect()

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_anom_file_path):
            print("## PKL Anomalous File: ", pkl_anom_file_path)
            anom_df = pd.read_pickle(pkl_anom_file_path)
        else:
            print("## Anomalous File: ", raw_normal_file_path)            
            anom_df = pd.read_csv(raw_anom_file_path, low_memory=True, header = 0, dtype=column_types)
            anom_df = data_cleasing(anom_df)
            anom_df.to_pickle(pkl_anom_file_path)
        gc.collect()

        # drop features
        normal_df.drop(drop_features[features_key], axis =1, inplace = True)
        anom_df.drop(drop_features[features_key], axis =1, inplace = True)

        # data splitting
        norm_train_df, cv_df, test_df, cv_label_df, test_label_df = data_splitting(normal_df, anom_df)

        # train
        norm_train_df.loc[:, 'Label'] = int(0)
        cv_df.loc[:, 'Label'] = cv_label_df
        train_df = pd.concat([norm_train_df, cv_df], axis=0)
        train_label_df = pd.concat([norm_train_df['Label'], cv_df['Label']], axis=0)
        train_df = train_df.drop(labels = ["Label"], axis = 1)
        
        best_f1_score = 0;
        best_batch_size = 0
        best_mbkmeans_pred_test_label = []
        test_label = test_label_df.astype(int).values

        for m_batch_size in range(10, 310, 10):
            
            mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=2, batch_size=m_batch_size, n_init=10, max_no_improvement=10)
            mbkmeans.fit(train_df)
            pred_test_label = mbkmeans.predict(test_df).astype(int)
            
            m_f1_score = f1_score(test_label, pred_test_label, average = "binary")
            if m_f1_score > best_f1_score:
                best_batch_size = m_batch_size
                best_f1_score = m_f1_score                
                best_mbkmeans_pred_test_label = pred_test_label
                print(best_batch_size, best_f1_score)

        mbkmeans_test_label.extend(test_label)
        mbkmeans_pred_test_label.extend(best_mbkmeans_pred_test_label)  # append into global array
        
    print('###', features_key, '[MBKMeans] Classification report for Test dataset')
    print_classification_report(mbkmeans_test_label, mbkmeans_pred_test_label)