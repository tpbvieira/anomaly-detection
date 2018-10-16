# coding=utf-8
'''
minibactch-kmeans implementation for train and test files
'''
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


# data cleasing, feature engineering and save clean data into pickles
def data_cleasing(df):
        
    # print('### Data Cleasing and Feature Engineering')
    le = preprocessing.LabelEncoder()
    
    df['Proto'] = df['Proto'].fillna('-')
    df['Proto'] = le.fit_transform(df['Proto'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    anomalies = df.Label.str.contains('Botnet')
    normal = np.invert(anomalies);
    df.loc[anomalies, 'Label'] = np.uint8(1)
    df.loc[normal, 'Label'] = np.uint8(0)
    df['Label'] = pd.to_numeric(df['Label'])

    df['Dport'] = df['Dport'].fillna('0')
    df['Dport'] = df['Dport'].apply(lambda x: int(x,0))

    try:
        df['Sport'] = df['Sport'].fillna('0')
        df['Sport'] = df['Sport'].str.replace('.*x+.*', '0')
        df['Sport'] = df['Sport'].apply(lambda x: int(x,0))
    except:                        
        print("Unexpected error:", sys.exc_info()[0])

    df['sTos'] = df['sTos'].fillna('10')
    df['sTos'] = df['sTos'].astype(int)

    df['dTos'] = df['dTos'].fillna('10')
    df['dTos'] = df['dTos'].astype(int)

    df['State'] = df['State'].fillna('-')
    df['State'] = le.fit_transform(df['State'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    df['Dir'] = df['Dir'].fillna('-')
    df['Dir'] = le.fit_transform(df['Dir'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    df['SrcAddr'] = df['SrcAddr'].fillna('0.0.0.0')
    
    df['DstAddr'] = df['DstAddr'].fillna('0.0.0.0')
    
    df['StartTime'] = df['StartTime'].apply(lambda x: x[:19])
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df = df.set_index('StartTime')
    
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

            for m_epsilon in np.arange(80, 95, 2):
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

# for each feature set
for features_key, value in drop_features.items():

    # Initialize labels
    mbkmeans_test_label = []
    mbkmeans_pred_test_label = []

    # Load data
    pkl_train_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl_all_train/train.binetflow'
    raw_train_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/raw_all_train/train.binetflow'
    pkl_test_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl_all_test/test.binetflow'
    raw_test_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/raw_all_test/test.binetflow'

    # read pickle or raw dataset for training
    if os.path.isfile(pkl_train_file_path):
        # print("### Train File: ", pkl_train_file_path)
        train_df = pd.read_pickle(pkl_train_file_path)
    else:
        # print("### Train File: ", raw_train_file_path)
        raw_df = pd.read_csv(raw_train_file_path, header = 0, dtype=column_types)
        train_df = data_cleasing(raw_df)
        # save clean data into pickles
        train_df.to_pickle(pkl_train_file_path)
    gc.collect()

    # read pickle or raw dataset for testing
    if os.path.isfile(pkl_test_file_path):
        # print("### Test File: ", pkl_test_file_path)
        test_df = pd.read_pickle(pkl_test_file_path)
    else:
        # print("### Test File: ", raw_test_file_path)
        raw_df = pd.read_csv(raw_test_file_path, header = 0, dtype=column_types)
        test_df = data_cleasing(raw_df)
        # save clean data into pickles
        test_df.to_pickle(pkl_test_file_path)
    gc.collect()
    
    # drop unnecessary features
    train_df.drop(drop_features[features_key], axis =1, inplace = True)
    test_df.drop(drop_features[features_key], axis =1, inplace = True)

    # data splitting
    train_len = (len(train_df) * 60) // 100
    cv_df = train_df[train_len+1:]                                      # use the last 40% of training data for cross-validation    
    train_df = train_df[:train_len]                                     # use the first 60% of training data for training
    train_df = train_df[train_df["Label"] == 0]                         # only normal data for training    
    train_df = train_df.drop(labels = ["Label"], axis = 1)              # drop label from training data
    cv_label_df = cv_df["Label"]                                        # save label for testing
    cv_df = cv_df.drop(labels = ["Label"], axis = 1)                    # drop label from testing data
    test_label_df = test_df["Label"]                                    # save label for testing
    test_df = test_df.drop(labels = ["Label"], axis = 1)                # drop label from testing data

    # Cross-Validation
    best_cluster_size, best_batch_size, best_epsilon, best_f1, best_precision, best_recall = getBestByCV(train_df, cv_df, cv_label_df)
    print('    ###[MB-KMeans][',features_key,'] Cross-Validation (cluster_size, batch_size, epsilon, f1, precision, recall): ', best_cluster_size, best_batch_size, best_epsilon, best_f1, best_precision, best_recall)

    # Training - estimate clusters (anomalous or normal) for training    
    mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=best_cluster_size, batch_size=best_batch_size, n_init=10, max_no_improvement=10).fit(train_df)
    
    # Test prediction
    test_clusters = mbkmeans.predict(test_df)
    test_clusters_centers = kmeans.cluster_centers_
    dist = [np.linalg.norm(x-y) for x,y in zip(test_df.as_matrix(), test_clusters_centers[test_clusters])]
    pred_test_label = np.array(dist)
    pred_test_label[dist >= np.percentile(dist, best_epsilon)] = 1
    pred_test_label[dist < np.percentile(dist, best_epsilon)] = 0

    # print results
    print('    ###[MB-KMeans][',features_key,'] Test')
    print_classification_report(test_label_df.astype(int).values, pred_test_label)