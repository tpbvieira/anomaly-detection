# coding=utf-8
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
        print('dropping ipv6 and icmp')
        df = df[df.Proto != 'ipv6']        
        df = df[df.Proto != 'ipv6-icmp']        
        df = df[df.Proto != 'icmp']
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])        

    try:
        print('Proto')
        df['Proto'] = df['Proto'].fillna('-')
        df['Proto'] = le.fit_transform(df['Proto'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['Proto'].head())
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
        print(df.head())
        print(df['Label'].head())
        df['Label'].to_csv('error_label.csv', index=False)

    try:
        print('Dport')
        df['Dport'] = df['Sport'].str.replace('.*x+.*', '0')
        df['Dport'] = df['Dport'].fillna('0')        
        df['Dport'] = df['Dport'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['Dport'].head())
        df['Dport'].to_csv('error_dport.csv', index=False)
    
    try:
        print('Sport')
        df['Sport'] = df['Sport'].str.replace('.*x+.*', '0')
        df['Sport'] = df['Sport'].fillna('0')        
        df['Sport'] = df['Sport'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info())
        print(df.head())
        print(df['Sport'].head())
        df['Sport'].to_csv('error_sport.csv', index=False)

    try:
        print('sTos')
        df['sTos'] = df['sTos'].fillna('10')
        df['sTos'] = df['sTos'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['Stos'].head())
        df['Stos'].to_csv('error_stos.csv', index=False)

    try:
        print('dTos')
        df['dTos'] = df['dTos'].fillna('10')
        df['dTos'] = df['dTos'].astype(int)
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['dTos'].head())
        df['dTos'].to_csv('error_dtos.csv', index=False)

    try:
        print('State')
        df['State'] = df['State'].fillna('-')
        df['State'] = le.fit_transform(df['State'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['State'].head())
        df['State'].to_csv('error_state.csv', index=False)

    try:
        print('Dir')
        df['Dir'] = df['Dir'].fillna('-')
        df['Dir'] = le.fit_transform(df['Dir'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['Dir'].head())
        df['Dir'].to_csv('error_dir.csv', index=False)

    try:
        print('SrcAddr')
        df['SrcAddr'] = df['SrcAddr'].fillna('0.0.0.0')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['SrcAddr'].head())
        df['SrcAddr'].to_csv('error_srcaddr.csv', index=False)
    
    try:
        print('DstAddr')
        df['DstAddr'] = df['DstAddr'].fillna('0.0.0.0')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['DstAddr'].head())
        df['DstAddr'].to_csv('error_dstaddr.csv', index=False)
    
    try:
        print('StartTime')
        df['StartTime'] = df['StartTime'].apply(lambda x: x[:19])
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df = df.set_index('StartTime')
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(df.head())
        print(df['StartTime'].head())
        df['StartTime'].to_csv('error_starttime.csv', index=False)
    
    return df

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_normal/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)

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

for sample_file in raw_files:
    raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
    print("### Reading: ", raw_file_path)    
    raw_df = pd.read_csv(raw_file_path, low_memory=True, header = 0, dtype=column_types)
    print("### Cleaning....", raw_df.shape)
    raw_df = data_cleasing(raw_df)
    print("### Clean!")
    gc.collect()