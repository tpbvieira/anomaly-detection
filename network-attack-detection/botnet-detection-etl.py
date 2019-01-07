# coding=utf-8
import os, sys, gc, time, warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing
warnings.filterwarnings("ignore")


def data_cleasing(m_df):

    # data cleasing, feature engineering and save clean data into pickles
    print('### Data Cleasing and Feature Engineering')
    le = preprocessing.LabelEncoder()

    # dropping ipv6 and icmp
    try:
        print('dropping ipv6 and icmp')
        m_df = m_df[m_df.Proto != 'ipv6']
        m_df = m_df[m_df.Proto != 'ipv6-icmp']
        m_df = m_df[m_df.Proto != 'icmp']
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])

    try:
        print('Proto')
        m_df['Proto'] = m_df['Proto'].fillna('-')
        m_df['Proto'] = le.fit_transform(m_df['Proto'])
        # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
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
        m_df['State'] = le.fit_transform(m_df['State'])
        # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        gc.collect()
    except:
        print(">>> Unexpected error:", sys.exc_info()[0])
        print(m_df.head())
        print(m_df['State'].head())
        m_df['State'].to_csv('error_state.csv', index=False)

    try:
        print('Dir')
        m_df['Dir'] = m_df['Dir'].fillna('-')
        m_df['Dir'] = le.fit_transform(m_df['Dir'])
        # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
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

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/raw/')
raw_directory = os.fsencode(raw_path)
file_list = os.listdir(raw_directory)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl/')
pkl_directory = os.fsencode(pkl_path)

# for each file
for sample_file in file_list:

    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(pkl_file_path):
        print("## PKL File Already Exist: ", pkl_file_path)
    else:  # load raw file and save clean data into pickles
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        print("## Raw File: ", raw_file_path)
        raw_df = pd.read_csv(raw_file_path, header=0, dtype=column_types)
        clean_df = data_cleasing(raw_df)
        clean_df.to_pickle(pkl_file_path)
    gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))