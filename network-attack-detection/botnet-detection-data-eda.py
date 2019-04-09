# coding=utf-8
import os
import gc
import warnings
import ipaddress
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from botnet_detection_utils import data_cleasing
warnings.filterwarnings(action='once')


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


raw_column_types = {
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
    'Label': 'str'
}

raw_features_names = [
    'Dur',
    'Proto',
    'Dir',
    'Dport',
    'Sport',
    'State',
    'TotPkts',
    'TotBytes'
]

agg_features_names = [
    'background_flow_count',
    'n_s_a_p_address',
    'avg_duration',
    'n_s_b_p_address',
    'n_sports<1024',
    'n_sports>1024',
    'n_conn',
    'n_s_na_p_address',
    'n_udp',
    'n_icmp',
    'n_d_na_p_address',
    'n_d_a_p_address',
    'n_s_c_p_address',
    'n_d_c_p_address',
    'normal_flow_count',
    'n_dports<1024',
    'n_d_b_p_address',
    'n_tcp'
]


# raw data
raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl')
raw_directory = os.fsencode(raw_path)
pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)
result_figures_path = 'results/raw/figures/'

# for each file (scenario)
for sample_file in file_list:

    # extract values from the file name
    file_name = os.path.splitext(sample_file.decode('utf-8'))[0]

    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(pkl_file_path):
        print("## Sample File: ", pkl_file_path)
        df = pd.read_pickle(pkl_file_path)
    else:  # load raw file and save clean data into pickles
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        print("## Sample File: ", raw_file_path)
        raw_df = pd.read_csv(raw_file_path, header=0, dtype=raw_column_types)
        df = data_cleasing(raw_df)
        df.to_pickle(pkl_file_path)
    gc.collect()

    # print("\n", df.head())
    # print("\nData Types: ", df.dtypes)
    # print("\nData Shape: ", df.shape)
    print("\nLabel:\n", df['Label'].value_counts())
    # print("\nProto:\n", df['Proto'].value_counts())
    # print("\nSrcAddr:\n", df['SrcAddr'].value_counts())
    # print("\nDstAddr:\n", df['DstAddr'].value_counts())
    # print("\nDport:\n", df['Dport'].value_counts())
    # print("\nState:\n", df['State'].value_counts())
    # print("\nsTos:\n", df['sTos'].value_counts())
    # print("\ndTos:\n", df['dTos'].value_counts())

    for i, cn in enumerate(df[raw_features_names]):
        fig_name = '%sraw_distplot_%s_%s.png' % (result_figures_path, file_name,cn)
        if not os.path.isfile(fig_name):
            sns.distplot(df[cn][df.Label == 1], bins=10, label='anomaly', color='r')
            sns.distplot(df[cn][df.Label == 0], bins=10, label='normal', color='b')
            plt.legend()
            plt.savefig(fig_name)
            plt.close()

    fig_name = '%sraw_corr_heatmap_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(15,10))
        sns_plot = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax, center=0, cmap="seismic")
        figure = sns_plot.get_figure()
        figure.savefig(fig_name, dpi=400)
        plt.close()

    # Create a pairplot for interesting features from heatmatp to
    fig_name = '%sraw_pairplot_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        plot_features = ['Dur','Proto','Dir','Sport','State','TotPkts','TotBytes']
        sns_plot = sns.pairplot(df, vars=plot_features, hue='Label')
        sns_plot.savefig(fig_name)
        plt.close()


# agg data
print('\n\n')
agg_path = os.path.join('/home/thiago/dev/anomaly-detection/network-attack-detection/data/ctu_13/pkl_sum')
agg_directory = os.fsencode(agg_path)
pkl_path = os.path.join('/home/thiago/dev/anomaly-detection/network-attack-detection/data/ctu_13/pkl_sum')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)
result_figures_path = 'results/agg/figures/'

# for each file (scenario)
for sample_file in file_list:

    # extract values from the file name
    file_name = os.path.splitext(sample_file.decode('utf-8'))[0]

    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(pkl_file_path):
        print("## Sample File: ", pkl_file_path)
        df = pd.read_pickle(pkl_file_path)
    else:  # load agg file and save clean data into pickles
        agg_file_path = os.path.join(agg_directory, sample_file).decode('utf-8')
        print("## Sample File: ", agg_file_path)
        agg_df = pd.read_csv(agg_file_path, header=0)
        df = data_cleasing(agg_df)
        df.to_pickle(pkl_file_path)
    gc.collect()

    # print("\n", df.head())
    # print("\nData Types: ", df.dtypes)
    # print("\nData Shape: ", df.shape)
    print("\nLabel:\n", df['Label'].value_counts())
    # print("\nbackground_flow_count:\n", df['background_flow_count'].value_counts())
    # print("\nn_s_a_p_address:\n", df['n_s_a_p_address'].value_counts())
    # print("\navg_duration:\n", df['avg_duration'].value_counts())
    # print("\nn_s_b_p_address:\n", df['n_s_b_p_address'].value_counts())
    # print("\nn_sports<1024:\n", df['n_sports<1024'].value_counts())
    # print("\nn_sports>1024:\n", df['n_sports>1024'].value_counts())
    # print("\nn_conn:\n", df['n_conn'].value_counts())
    # print("\nn_s_na_p_address:\n", df['n_s_na_p_address'].value_counts())
    # print("\nn_udp:\n", df['n_udp'].value_counts())
    # print("\nn_icmp:\n", df['n_icmp'].value_counts())
    # print("\nn_d_na_p_address:\n", df['n_d_na_p_address'].value_counts())
    # print("\nn_d_a_p_address:\n", df['n_d_a_p_address'].value_counts())
    # print("\nn_s_c_p_address:\n", df['n_s_c_p_address'].value_counts())
    # print("\nn_d_c_p_address:\n", df['n_d_c_p_address'].value_counts())
    # print("\nnormal_flow_count:\n", df['normal_flow_count'].value_counts())
    # print("\nn_dports<1024:\n", df['n_dports<1024'].value_counts())
    # print("\nn_d_b_p_address:\n", df['n_d_b_p_address'].value_counts())
    # print("\nn_tcp:\n", df['n_tcp'].value_counts())

    for i, cn in enumerate(df[agg_features_names]):
        fig_name = '%sagg_distplot_%s_%s.png' % (result_figures_path, file_name,cn)
        if not os.path.isfile(fig_name):
            sns.distplot(df[cn][df.Label == 1], bins=10, label='anomaly', color='r')
            sns.distplot(df[cn][df.Label == 0], bins=10, label='normal', color='b')
            plt.legend()
            plt.savefig(fig_name)
            plt.close()

    fig_name = '%sagg_corr_heatmap_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(15,10))
        sns_plot = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax, center=0, cmap="seismic")
        figure = sns_plot.get_figure()
        figure.savefig(fig_name, dpi=400)
        plt.close()

    # # Create a pairplot for interesting features from heatmatp to
    # fig_name = '%sagg_pairplot_%s.png' % (result_figures_path, file_name)
    # if not os.path.isfile(fig_name):
    #     plot_features = ['Dur','Proto','Dir','Sport','State','TotPkts','TotBytes']
    #     sns_plot = sns.pairplot(df, vars=plot_features, hue='Label')
    #     sns_plot.savefig(fig_name)
    #     plt.close()

# # The datastructure to hold our feature extraction functions,
# # which will get applied to each aggregation of the datasets.
# extractors = {
#     'Label': [label_atk_v_norm, background_flow_count, normal_flow_count, n_conn, ],
#     'Dport': [n_dports_gt1024, n_dports_lt1024],
#     'Sport': [n_sports_gt1024, n_sports_lt1024, ],
#     'Dur': [avg_duration, ],
#     'SrcAddr': [n_s_a_p_address, n_s_b_p_address, n_s_c_p_address, n_s_na_p_address, ],
#     'DstAddr': [n_d_a_p_address, n_d_b_p_address, n_d_c_p_address, n_d_na_p_address, ],
#     'Proto': [n_tcp, n_icmp, n_udp, ],
# }
#
# # resample grouped by 1 second bin. must have a datetime-like index.
# r = df.resample('1S')
# n_df = r.agg(extractors)  ## aggretation by data and functions specified by extractors
#
# n_df.columns = n_df.columns.droplevel(0)  # get rid of the heirarchical columns
# pd.options.display.max_columns = 99
#
#
# print('New nData Types: ', n_df.dtypes)
# print(n_df.head())

#
# fig_name = '%ssum_distplot.png'
# if not os.path.isfile(fig_name):
#     n_features = ['background_flow_count','normal_flow_count','n_conn','n_dports>1024','n_dports<1024','n_s_a_p_address','n_s_b_p_address','n_s_c_p_address','n_s_na_p_address','n_d_a_p_address','n_d_b_p_address','n_d_c_p_address']
#     nplots=np.size(n_features)
#     plt.figure(figsize=(15,4*nplots))
#     gs = gridspec.GridSpec(nplots,1)
#     for i, cn in enumerate(n_df[n_features]):
#         try:
#             ax = plt.subplot(gs[i])
#             sns.distplot(n_df[cn][n_df.label == 1], bins=10, label='anomaly', color='r')
#             sns.distplot(n_df[cn][n_df.label == 0], bins=10, label='normal', color='b')
#             ax.set_xlabel('')
#             ax.set_title('feature: ' + str(cn))
#             plt.legend()
#         except Exception as e:
#             print(e)
#     plt.savefig(fig_name)
#
# fig_name = '%ssum_corr_heatmap.png'
# if not os.path.isfile(fig_name):
#     corr = n_df.corr()
#     fig, ax = plt.subplots(figsize=(15,10))
#     sns_plot = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax, center=0, cmap="seismic")
#     figure = sns_plot.get_figure()
#     figure.savefig(fig_name, dpi=400)
#
#
# # drops N/A values
# tmp_df = n_df.dropna()
#
# # Create a scatter matrix of the aggregated dataframe
# # choose a few interesting features to pairplot based on the heat maps
# fig_name = '%ssum_pairplot.png'
# if not os.path.isfile(fig_name):
#     plot_features = ['avg_duration','n_udp','background_flow_count','n_conn','n_icmp']
#     sns_plot = sns.pairplot(tmp_df, vars=plot_features, hue='label')
#     sns_plot.savefig(fig_name)
#
#
# # drops N/A values
# tmp_df = n_df.dropna()
#
# # Create a scatter matrix of the aggregated dataframe
# # choose a few interesting features to pairplot based on the heat maps
# fig_name = '%ssum_pairplot2.png'
# if not os.path.isfile(fig_name):
#     plot_features = ['avg_duration','n_conn']
#     sns_plot = sns.pairplot(tmp_df, vars=plot_features, hue='label')
#     sns_plot.savefig(fig_name)