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
    # data cleasing, feature engineering and save clean data into pickles
    
    print('### Data Cleasing and Feature Engineering')
    le = preprocessing.LabelEncoder()
    
    # [Protocol] - Discard ipv6-icmp and categorize
    # df = df[df.Proto != 'ipv6-icmp']
    df['Proto'] = df['Proto'].fillna('-')
    df['Proto'] = le.fit_transform(df['Proto'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # [Label] - Categorize 
    anomalies = df.Label.str.contains('Botnet')
    normal = np.invert(anomalies);
    df.loc[anomalies, 'Label'] = np.uint8(1)
    df.loc[normal, 'Label'] = np.uint8(0)
    df['Label'] = pd.to_numeric(df['Label'])

    # [Dport] - replace NaN with 0 port number
    df['Dport'] = df['Dport'].fillna('0')
    df['Dport'] = df['Dport'].apply(lambda x: int(x,0))

    # [sport] - replace NaN with 0 port number
    try:
        df['Sport'] = df['Sport'].fillna('0')
        df['Sport'] = df['Sport'].str.replace('.*x+.*', '0')
        df['Sport'] = df['Sport'].apply(lambda x: int(x,0))
    except:                        
        print("Unexpected error:", sys.exc_info()[0])

    # [sTos] - replace NaN with "10" and convert to int
    df['sTos'] = df['sTos'].fillna('10')
    df['sTos'] = df['sTos'].astype(int)

    # [dTos] - replace NaN with "10" and convert to int
    df['dTos'] = df['dTos'].fillna('10')
    df['dTos'] = df['dTos'].astype(int)

    # [State] - replace NaN with "-" and categorize
    df['State'] = df['State'].fillna('-')
    df['State'] = le.fit_transform(df['State'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # [Dir] - replace NaN with "-" and categorize 
    df['Dir'] = df['Dir'].fillna('-')
    df['Dir'] = le.fit_transform(df['Dir'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # [SrcAddr] Extract subnet features and categorize
    df['SrcAddr'] = df['SrcAddr'].fillna('0.0.0.0')
    # tmp_df = pd.DataFrame(df['SrcAddr'].str.split('.').tolist(), columns = ['1','2','3','4'])
    # df["SrcAddr1"] = tmp_df["1"]
    # df["SrcAddr2"] = tmp_df["1"].map(str) + tmp_df["2"]
    # df["SrcAddr3"] = tmp_df["1"].map(str) + tmp_df["2"].map(str) + tmp_df["3"]
    # df['SrcAddr0'] = le.fit_transform(df['SrcAddr'])
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # df['SrcAddr1'] = df['SrcAddr1'].astype(int)
    # df['SrcAddr2'] = df['SrcAddr2'].astype(int)
    # df['SrcAddr3'] = df['SrcAddr3'].astype(int)

    # [DstAddr] Extract subnet features
    df['DstAddr'] = df['DstAddr'].fillna('0.0.0.0')
    # tmp_df = pd.DataFrame(df['DstAddr'].str.split('.').tolist(), columns = ['1','2','3','4'])
    # df["DstAddr1"] = tmp_df["1"]
    # df["DstAddr2"] = tmp_df["1"].map(str) + tmp_df["2"]
    # df["DstAddr3"] = tmp_df["1"].map(str) + tmp_df["2"].map(str) + tmp_df["3"]
    # df['DstAddr0'] = le.fit_transform(df['DstAddr'])
    # df['DstAddr1'] = df['DstAddr1'].astype(int)
    # df['DstAddr2'] = df['DstAddr2'].astype(int)
    # df['DstAddr3'] = df['DstAddr3'].astype(int)

    # [StartTime] - Parse to datatime, reindex based on StartTime, but first drop the ns off the time stamps
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

def selectThresholdByCV(probs, labels):
    # select best epsilon (threshold)
    
    # initialize
    best_epsilon = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    min_prob = min(probs);
    max_prob = max(probs);
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)
    
    for epsilon in epsilons:
        predictions = (probs < epsilon)
        
        f1 = f1_score(labels, predictions, average = "binary")
        Recall = recall_score(labels, predictions, average = "binary")
        Precision = precision_score(labels, predictions, average = "binary")            
        
        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
            best_precision = Precision
            best_recall = Recall

    return best_f1, best_epsilon


def print_classification_report(y_test, y_predic):
    print('### Classification report:')
    print(classification_report(y_test, y_predic))

    print('\tAverage Precision = ' + str(average_precision_score(y_test, y_predic)))

    print('\n### Binary F1 Score, Recall and Precision:')
    f = f1_score(y_test, y_predic, average = "binary")
    Recall = recall_score(y_test, y_predic, average = "binary")
    Precision = precision_score(y_test, y_predic, average = "binary")
    print('\tF1 Score %f' %f)
    print('\tRecall Score %f' %Recall)
    print('\tPrecision Score %f' %Precision)


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

def data_splitting(df):
    # Data splitting
    
    # drop non discriminant features
    df.drop([
    #             'Dur',
                'Proto',
                'SrcAddr',
                # 'Sport',
    #             'Dir',
                'DstAddr',
    #             'Dport',
    #             'State',
                'sTos'
    #             'dTos',
    #             'TotPkts',
    #             'TotBytes',
                # 'SrcBytes'
    ], axis =1, inplace = True)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]

    # Length and indexes
    anom_len = len(df_l1)                                                # total number of anomalous flows
    anom_train_end = anom_len // 2                                # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1                        # 50% of anomalous for testing
    norm_len = len(df_l0)                                                # total number of normal flows
    norm_train_end = (norm_len * 60) // 100                # 60% of normal for training
    norm_cv_start = norm_train_end + 1                        # 20% of normal for cross validation
    norm_cv_end = (norm_len * 80) // 100                # 20% of normal for cross validation
    norm_test_start = norm_cv_end + 1                        # 20% of normal for testing

    # anomalies split data
    anom_cv_df  = df_l1[:anom_train_end]                         # 50% of anomalies59452 
    anom_test_df = df_l1[anom_cv_start:anom_len]         # 50% of anomalies

    # normal split data
    norm_train_df = df_l0[:norm_train_end]                                 # 60% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]                 # 20% of normal
    norm_test_df = df_l0 [norm_test_start:norm_len]         # 20% of normal

    # CV and test data. train data is norm_train_df
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)

    # labels
    cv_label = cv_df["Label"]
    test_label = test_df["Label"]

    # drop label
    norm_train_df = norm_train_df.drop(labels = ["Label"], axis = 1)
    cv_df = cv_df.drop(labels = ["Label"], axis = 1)
    test_df = test_df.drop(labels = ["Label"], axis = 1)
    
    return norm_train_df, cv_df, test_df, cv_label, test_label

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

# Get raw file path
raw_all_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_all/')
raw_all_directory = os.fsencode(raw_all_path)
raw_all_files = os.listdir(raw_all_directory)
print("Directory: ", raw_all_directory)
print("Files: ", raw_all_files)

# Get pickle file path
pkl_all_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/pkl_all/')
pkl_all_directory = os.fsencode(pkl_all_path)

kmeans_test_label = []
kmeans_pred_test_label = []

# Load data
for sample_file in raw_all_files:
    pkl_all_file_path = os.path.join(pkl_all_directory, sample_file).decode('utf-8')
    raw_all_file_path = os.path.join(raw_all_directory, sample_file).decode('utf-8')

    # read pickle or raw dataset file with pandas
    if os.path.isfile(pkl_all_file_path):
        print("## Sample File: ", pkl_all_file_path)
        df = pd.read_pickle(pkl_all_file_path)
    else:
        print("## Sample File: ", raw_all_file_path)
        raw_df = pd.read_csv(raw_all_file_path, header = 0, dtype=column_types)
        df = data_cleasing(raw_df)
        # save clean data into pickles
        df.to_pickle(pkl_all_file_path)  # where to save it, usually as a .pkl
    gc.collect()
                
    # data splitting
    norm_train_df, cv_df, test_df, cv_label, test_label = data_splitting(df)
    norm_train_df.loc[:, 'Label'] = int(0)
    cv_df.loc[:, 'Label'] = cv_label

    # train data from concatenation 
    train_df = pd.concat([norm_train_df, cv_df], axis=0)                                                        # concatenate normal and cross validation (normal and anomalous) data
    train_label_df = pd.concat([norm_train_df['Label'], cv_df['Label']], axis=0)        # concatenate normal and cross validation (normal and anomalous) label
    train_df = train_df.drop(labels = ["Label"], axis = 1)                                                        # drop label from train dat
    
    # Training - estimate clusters (anomalous or normal) for training
    mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=2, batch_size=40, n_init=10, max_no_improvement=10)
    mbkmeans.fit(train_df)
    
    # Testing
    pred_test_label = mbkmeans.predict(test_df)
    kmeans_test_label.extend(test_label.astype(int))                                # append into global array
    kmeans_pred_test_label.extend(pred_test_label.astype(int))                # append into global array

# save results
np.savetxt('output/mbkmeans_test_label.out', kmeans_test_label, delimiter=',')
np.savetxt('output/mnkmeans_pred_test_label.out', kmeans_pred_test_label, delimiter=',')
            
# print results
print ('\n[MBKMeans] Classification report for Cross Validation dataset')
print_classification_report(kmeans_test_label, kmeans_pred_test_label)