# coding=utf-8
import os
import gc
import warnings
import ipaddress
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from functools import reduce
from botnet_detection_utils import ctu13_data_cleasing
warnings.filterwarnings(action='once')


def get_best_distribution_fit(data):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    dist_names = ['beta',
                  'expon',
                  'gamma',
                  'lognorm',
                  'norm',
                  'pearson3',
                  'triang',
                  'uniform',
                  'weibull_min',
                  'weibull_max']

    # Set up empty lists to stroe results
    chi_square = []
    p_values = []

    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distributed across all bins
    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(data, percentile_bins)
    observed_frequency, bins = (np.histogram(data, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(stats, distribution)
        param = dist.fit(data)

        # Obtain the KS test P statistic, round it to 5 decimal places
        p = stats.kstest(data, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)

        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * data.shape[0]
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)

    # Collate results and sort by goodness of fit (best at top)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)

    return results


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
raw_path = os.path.join('data/ctu_13/raw/')
raw_directory = os.fsencode(raw_path)
raw_pkl_path = os.path.join('data/ctu_13/raw_clean_pkl/')
raw_pkl_directory = os.fsencode(raw_pkl_path)
file_list = os.listdir(raw_pkl_directory)
result_figures_path = 'output/ctu_13/eda/raw/figures/'
raw_eda_file_path = 'output/ctu_13/eda/raw/raw_eda.txt'

raw_eda_file = open(raw_eda_file_path, 'w') # 'w' = clear all and write
# for each raw scenario
for sample_file in file_list:

    # extract values from the file name
    file_name = os.path.splitext(sample_file.decode('utf-8'))[0]

    # read pickle file with pandas or...
    raw_pkl_file_path = os.path.join(raw_pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(raw_pkl_file_path):
        print("## Sample File: %s" % raw_pkl_file_path, file=raw_eda_file)
        df = pd.read_pickle(raw_pkl_file_path)
    else:  # load raw file and save clean data into pickles
        raw_eda_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        print("## Sample File: %s" % raw_eda_file_path, file=raw_eda_file)
        raw_df = pd.read_csv(raw_eda_file_path, header=0, dtype=raw_column_types)
        df = ctu13_data_cleasing(raw_df)
        df.to_pickle(raw_pkl_file_path)
    gc.collect()
    raw_eda_file.flush()

    print("\n", df.head(), file=raw_eda_file)
    print("\nData Types: ", df.dtypes, file=raw_eda_file)
    print("\nData Shape: ", df.shape, file=raw_eda_file)
    print("\nLabel:\n", df['Label'].value_counts(), file=raw_eda_file)
    print("\nProto:\n", df['Proto'].value_counts(), file=raw_eda_file)
    print("\nSrcAddr:\n", df['SrcAddr'].value_counts(), file=raw_eda_file)
    print("\nDstAddr:\n", df['DstAddr'].value_counts(), file=raw_eda_file)
    print("\nDport:\n", df['Dport'].value_counts(), file=raw_eda_file)
    print("\nState:\n", df['State'].value_counts(), file=raw_eda_file)
    print("\nsTos:\n", df['sTos'].value_counts(), file=raw_eda_file)
    print("\ndTos:\n", df['dTos'].value_counts(), file=raw_eda_file)
    raw_eda_file.flush()

    # # Print the distribution fitting for each feature
    # for i, cn in enumerate(df[raw_features_names]):
    #     best_dist_fitting = get_best_distribution_fit(df[cn])
    #     print('###', cn, file=raw_eda_file)
    #     print(best_dist_fitting, file=raw_eda_file)
    #     raw_eda_file.flush()
    #
    # # Plot the distribution of each feature
    # for i, cn in enumerate(df[raw_features_names]):
    #     fig_name = '%sraw_distplot_%s_%s.png' % (result_figures_path, file_name,cn)
    #     if not os.path.isfile(fig_name):
    #         sns.distplot(df[cn][df.Label == 1], bins=10, label='anomaly', color='r')
    #         sns.distplot(df[cn][df.Label == 0], bins=10, label='normal', color='b')
    #         plt.legend()
    #         plt.savefig(fig_name)
    #         plt.close()

    # Plot a correlation heatmap of all features, including the label
    fig_name = '%sraw_corr_heatmap_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(15,10))
        sns_plot = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax, center=0,
                               cmap="seismic")
        figure = sns_plot.get_figure()
        figure.savefig(fig_name, dpi=400)
        plt.close()

    # Create scatter and histogram to show the relationship of interesting features (selected from heatmap)
    fig_name = '%sraw_pairplot_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        plot_features = ['Dur','Proto','Dir','Sport','State','TotPkts','TotBytes']
        sns_plot = sns.pairplot(df, vars=plot_features, hue='Label')
        sns_plot.savefig(fig_name)
        plt.close()
raw_eda_file.close()

# agg data
agg_path = os.path.join('data/ctu_13/agg_pkl/')
agg_directory = os.fsencode(agg_path)
agg_pkl_path = os.path.join('data/ctu_13/agg_pkl/')
agg_pkl_directory = os.fsencode(agg_pkl_path)
file_list = os.listdir(agg_pkl_directory)
result_figures_path = 'output/ctu_13/eda/agg/figures/'
agg_eda_file_path = 'output/ctu_13/eda/agg/agg_eda.txt'

agg_eda_file = open(agg_eda_file_path, 'w') # 'w' = clear all and write
print('\n\n', file=agg_eda_file)
# for each file (scenario)
for sample_file in file_list:

    # extract values from the file name
    file_name = os.path.splitext(sample_file.decode('utf-8'))[0]

    # read pickle file with pandas or...
    agg_pkl_file_path = os.path.join(agg_pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(agg_pkl_file_path):
        print("## Sample File: %s" % agg_pkl_file_path, file=agg_eda_file)
        df = pd.read_pickle(agg_pkl_file_path)
    else:  # load agg file and save clean data into pickles
        agg_file_path = os.path.join(agg_directory, sample_file).decode('utf-8')
        print("## Sample File: %s" % agg_file_path, file=agg_eda_file)
        agg_df = pd.read_csv(agg_file_path, header=0)
        df = ctu13_data_cleasing(agg_df)
        df.to_pickle(agg_pkl_file_path)
    gc.collect()
    agg_eda_file.flush()

    print("\n", df.head(), file=agg_eda_file)
    print("\nData Types: ", df.dtypes, file=agg_eda_file)
    print("\nData Shape: ", df.shape, file=agg_eda_file)
    print("\nLabel:\n", df['Label'].value_counts(), file=agg_eda_file)
    print("\nflow_count:\n", df['flow_count'].value_counts(), file=agg_eda_file)
    print("\nn_s_a_p_address:\n", df['n_s_a_p_address'].value_counts(), file=agg_eda_file)
    print("\navg_duration:\n", df['avg_duration'].value_counts(), file=agg_eda_file)
    print("\nn_s_b_p_address:\n", df['n_s_b_p_address'].value_counts(), file=agg_eda_file)
    print("\nn_sports<1024:\n", df['n_sports<1024'].value_counts(), file=agg_eda_file)
    print("\nn_sports>1024:\n", df['n_sports>1024'].value_counts(), file=agg_eda_file)
    print("\nn_conn:\n", df['n_conn'].value_counts(), file=agg_eda_file)
    print("\nn_s_na_p_address:\n", df['n_s_na_p_address'].value_counts(), file=agg_eda_file)
    print("\nn_udp:\n", df['n_udp'].value_counts(), file=agg_eda_file)
    print("\nn_icmp:\n", df['n_icmp'].value_counts(), file=agg_eda_file)
    print("\nn_d_na_p_address:\n", df['n_d_na_p_address'].value_counts(), file=agg_eda_file)
    print("\nn_d_a_p_address:\n", df['n_d_a_p_address'].value_counts(), file=agg_eda_file)
    print("\nn_s_c_p_address:\n", df['n_s_c_p_address'].value_counts(), file=agg_eda_file)
    print("\nn_d_c_p_address:\n", df['n_d_c_p_address'].value_counts(), file=agg_eda_file)
    print("\nnormal_flow_count:\n", df['normal_flow_count'].value_counts(), file=agg_eda_file)
    print("\nn_dports<1024:\n", df['n_dports<1024'].value_counts(), file=agg_eda_file)
    print("\nn_d_b_p_address:\n", df['n_d_b_p_address'].value_counts(), file=agg_eda_file)
    print("\nn_tcp:\n", df['n_tcp'].value_counts(), file=agg_eda_file)
    agg_eda_file.flush()

    # # Print the distribution fitting for each feature
    # for i, cn in enumerate(df[agg_features_names]):
    #     best_dist_fitting = get_best_distribution_fit(df[cn])
    #     print('###', cn, file=agg_eda_file)
    #     print(best_dist_fitting, file=agg_eda_file)
    #     agg_eda_file.flush()
    #
    # # Plot the distribution of each feature
    # for i, cn in enumerate(df[agg_features_names]):
    #     fig_name = '%sagg_distplot_%s_%s.png' % (result_figures_path, file_name,cn)
    #     if not os.path.isfile(fig_name):
    #         sns.distplot(df[cn][df.Label == 1], bins=10, label='anomaly', color='r')
    #         sns.distplot(df[cn][df.Label == 0], bins=10, label='normal', color='b')
    #         plt.legend()
    #         plt.savefig(fig_name)
    #         plt.close()

    # Plot a correlation heatmap of all features, including the label
    fig_name = '%sagg_corr_heatmap_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(15,10))
        sns_plot = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax, center=0,
                               cmap="seismic")
        figure = sns_plot.get_figure()
        figure.savefig(fig_name, dpi=400)
        plt.close()

    # Create scatter and histogram to show the relationship of interesting features (selected from heatmap)
    fig_name = '%sagg_pairplot_%s.png' % (result_figures_path, file_name)
    if not os.path.isfile(fig_name):
        plot_features = ['flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address', 'n_sports<1024',
                         'n_s_na_p_address', 'n_s_c_p_address', 'n_d_c_p_address', 'normal_flow_count', 'n_dports<1024']
        sns_plot = sns.pairplot(df, vars=plot_features, hue='Label')
        sns_plot.savefig(fig_name)
        plt.close()
agg_eda_file.close()
