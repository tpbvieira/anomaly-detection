# coding=utf-8
import os, warnings, pickle, math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")


def get_outliers(m_dist, m_best_contamination):
    m_pred_label = np.full(m_dist.shape[0], 0, dtype=int)
    m_best_contamination = np.percentile(m_dist, 100. * m_best_contamination)
    m_pred_label[m_dist <= m_best_contamination] = 1

    # n, self.degrees_of_freedom_ = X.shape
    # self.iextreme_values = (self.d2 > self.chi2.ppf(0.995, self.degrees_of_freedom_))

    return m_pred_label


# result file path
pkl_path = os.path.join('results/pkl_sum_dict/20/data/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
columns = ['f1', 'approaches', 'algs', 'windows', 'scenarios']
result_df = pd.DataFrame(columns=columns)

# for each scenario (file name)
for sample_file in file_list:

    # extract values from the file name
    file_name = os.path.splitext(sample_file.decode('utf-8'))[0]
    tokens = file_name.split('_')
    it = tokens[1]
    window = tokens[2]
    scenario = tokens[3]

    # read the file
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    pkl_file = open(pkl_file_path, 'rb')
    result_dict = pickle.load(pkl_file)
    pkl_file.close()

    # print data overview
    test_label_series = result_dict.get(1).get('test_label')
    best_cont = result_dict.get(1).get('best_contamination')
    best_train_f1 = result_dict.get(1).get('training_f1')
    counts = test_label_series.value_counts()
    total_counts =  len(test_label_series)
    print('###[botnet-detection-results-eda] ', sample_file)
    print("\tTotal:", total_counts, ", 0:", counts.get(0),'(', round((counts.get(0)/total_counts)*100, 2),'%), 1:',
          counts.get(1),'(', round((counts.get(1)/total_counts)*100, 2),'%), best_contamination:', best_cont,
          ', best_train_f1:', best_train_f1)

    # for each iteration compute F1 and save the algorithm, window size, scenario and approach (window size and algorithm))
    for key, value in result_dict.items():

        # get saved data
        best_contamination = result_dict.get(key).get('best_contamination')
        mcd_pred_label = result_dict.get(key).get('mcd_test_label')
        k_pred_label = result_dict.get(key).get('k_test_label')
        s_pred_label = result_dict.get(key).get('s_test_label')
        test_label = result_dict.get(key).get('test_label')
        mcd_pred_dist_ = result_dict.get(key).get('mcd_prediction_dist_')
        k_pred_dist_ = result_dict.get(key).get('k_prediction_dist_')
        s_pred_dist_ = result_dict.get(key).get('s_prediction_dist_')

        # combine distances
        mks_dist_ = mcd_pred_dist_ - k_pred_dist_ - s_pred_dist_
        mk_dist_ = mcd_pred_dist_ - k_pred_dist_
        ms_dist_ = mcd_pred_dist_ - s_pred_dist_
        ks_dist_ = k_pred_dist_ + s_pred_dist_

        # get outliers from new distances
        m_label = get_outliers(-mcd_pred_dist_, best_contamination)
        k_pred_label = get_outliers(k_pred_dist_, best_contamination)
        s_pred_label = get_outliers(s_pred_dist_, best_contamination)
        mks_label = get_outliers(mks_dist_, best_contamination)
        mk_label = get_outliers(mk_dist_, best_contamination)
        ms_label = get_outliers(ms_dist_, best_contamination)
        ks_label = get_outliers(ks_dist_, best_contamination)

        # compute F1 score and append results into a mcd dataframe
        m_pred_f1 = f1_score(mcd_pred_label, test_label, average="binary")
        alg = 'mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m_pred_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a k-mcd dataframe
        k_pred_f1 = f1_score(k_pred_label, test_label, average="binary")
        alg = 'k-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[k_pred_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a s-mcd dataframe
        s_pred_f1 = f1_score(s_pred_label, test_label, average="binary")
        alg = 's-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[s_pred_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a mcd dataframe
        m_f1 = f1_score(m_label, test_label, average="binary")
        alg = 'sv-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a s-mcd dataframe
        mks_f1 = f1_score(mks_label, test_label, average="binary")
        alg = 'mks-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[mks_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a s-mcd dataframe
        mk_f1 = f1_score(mk_label, test_label, average="binary")
        alg = 'mk-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[mk_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a s-mcd dataframe
        ms_f1 = f1_score(ms_label, test_label, average="binary")
        alg = 'ms-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[ms_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a s-mcd dataframe
        ks_f1 = f1_score(ks_label, test_label, average="binary")
        alg = 'ks-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[ks_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        # # only for the first iteration, plot distances into boxplot and pairplot
        # if key == 0:
        #
        #     m_columns = ['mcd_dist_', 'k_dist_', 's_dist_', 'mks_dist_', 'mk_dist_',
        #                  'ms_dist_', 'ks_dist_', 'test_label']
        #     result_scenario_df = pd.DataFrame(list(zip(mcd_pred_dist_, k_pred_dist_, s_pred_dist_, mks_dist_, mk_dist_,
        #                                                ms_dist_, ks_dist_, test_label)), columns=m_columns)
        #
        #     # boxplot mcd_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="mcd_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_mcd_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t',fig_name)
        #
        #     # boxplot k_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="k_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_k_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t',fig_name)
        #
        #     # boxplot s_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="s_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_s_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t',fig_name)
        #
        #     # boxplot mks_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="mks_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_mks_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t', fig_name)
        #
        #     # boxplot mk_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="mk_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_mk_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t', fig_name)
        #
        #     # boxplot ms_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="ms_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_ms_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t', fig_name)
        #
        #     # boxplot ks_dist_
        #     plt.figure(figsize=(15, 5))
        #     bp = sns.boxplot(x="test_label", y="ks_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
        #     bp.set(yscale="log")
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_ks_dist.png" % (scenario, window)
        #     plt.savefig(fig_name)
        #     plt.close()
        #     print('\t', fig_name)
        #
        #     # pairplot
        #     plot_features = [ 'mks_dist_', 'mk_dist_', 'ms_dist_', 'ks_dist_']
        #     sns_distplot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label', diag_kind = 'kde',
        #      plot_kws = {'alpha': 0.1, 's': 80, 'edgecolor': 'k'}, size = 4)
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_distances_new.png" % (scenario, window)
        #     sns_distplot.savefig(fig_name)
        #     print('\t',fig_name)
        #
        #     plot_features = ['mcd_dist_', 'k_dist_', 's_dist_']
        #     sns_distplot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label', diag_kind='kde',
        #                                 plot_kws={'alpha': 0.1, 's': 80, 'edgecolor': 'k'}, size=4)
        #     fig_name = "results/pkl_sum_dict/20/figures/%s_%s_distances.png" % (scenario, window)
        #     sns_distplot.savefig(fig_name)
        #     print('\t', fig_name)

# for each scenario (original file name, such as 10 or 18-2)
m_scenarios = result_df.scenarios.unique()
for m_scenario in m_scenarios:

    # get data by scenario and group by window and algorithm
    result_scenario_df = result_df.loc[result_df['scenarios'] == m_scenario]
    result_scenario_df = result_scenario_df.sort_values(['windows', 'algs'], ascending=[True, False])

    # boxplot F1 score by approaches for this scenario
    plt.figure(figsize=(15, 5))
    bp = sns.boxplot(x="approaches", y="f1", data=result_scenario_df, palette="PRGn", width=0.4)
    fig_name = "results/pkl_sum_dict/20/figures/%s.png" % m_scenario
    plt.savefig(fig_name)
    plt.close()

    # get median F1 from each approach
    m1 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_s-mcd'].median().get('f1')
    m2 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_s-mcd'].median().get('f1')
    m3 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_s-mcd'].median().get('f1')
    m4 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_s-mcd'].median().get('f1')
    m5 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_k-mcd'].median().get('f1')
    m6 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_k-mcd'].median().get('f1')
    m7 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_k-mcd'].median().get('f1')
    m8 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_k-mcd'].median().get('f1')
    m9 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_mcd'].median().get('f1')
    m10 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_mcd'].median().get('f1')
    m11 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_mcd'].median().get('f1')
    m12 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_mcd'].median().get('f1')
    m13 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_mks-mcd'].median().get('f1')
    m14 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_mks-mcd'].median().get('f1')
    m15 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_mks-mcd'].median().get('f1')
    m16 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_mks-mcd'].median().get('f1')
    m17 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_mk-mcd'].median().get('f1')
    m18 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_mk-mcd'].median().get('f1')
    m19 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_mk-mcd'].median().get('f1')
    m20 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_mk-mcd'].median().get('f1')
    m21 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_ms-mcd'].median().get('f1')
    m22 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_ms-mcd'].median().get('f1')
    m23 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_ms-mcd'].median().get('f1')
    m24 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_ms-mcd'].median().get('f1')
    m25 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_ks-mcd'].median().get('f1')
    m26 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_ks-mcd'].median().get('f1')
    m27 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_ks-mcd'].median().get('f1')
    m28 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_ks-mcd'].median().get('f1')
    m29 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_sv-mcd'].median().get('f1')
    m30 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_sv-mcd'].median().get('f1')
    m31 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_sv-mcd'].median().get('f1')
    m32 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_sv-mcd'].median().get('f1')

    # detect the approach with the largest F1 score for the current scenario and print it
    f1_max = 0
    best_method = ''
    if not math.isnan(m1) and m1 > f1_max:
        f1_max = m1
        best_method = '0.15s_s-mcd'
    if not math.isnan(m2) and m2 > f1_max:
        f1_max = m2
        best_method = '0.25s_s-mcd'
    if not math.isnan(m3) and m3 > f1_max:
        f1_max = m3
        best_method = '1s_s-mcd'
    if not math.isnan(m4) and m4 > f1_max:
        f1_max = m4
        best_method = '2s_s-mcd'
    if not math.isnan(m5) and m5 > f1_max:
        f1_max = m5
        best_method = '0.15s_k-mcd'
    if not math.isnan(m6) and m6 > f1_max:
        f1_max = m6
        best_method = '0.25s_k-mcd'
    if not math.isnan(m7) and m7 > f1_max:
        f1_max = m7
        best_method = '1s_k-mcd'
    if not math.isnan(m8) and m8 > f1_max:
        f1_max = m8
        best_method = '2s_k-mcd'
    if not math.isnan(m9) and m9 > f1_max:
        f1_max = m9
        best_method = '0.15s_mcd'
    if not math.isnan(m10) and m10 > f1_max:
        f1_max = m10
        best_method = '0.25s_mcd'
    if not math.isnan(m11) and m11 > f1_max:
        f1_max = m11
        best_method = '1s_mcd'
    if not math.isnan(m12) and m12 > f1_max:
        f1_max = m12
        best_method = '2s_mcd'
    if not math.isnan(m13) and m13 > f1_max:
        f1_max = m13
        best_method = '0.15s_mks-mcd'
    if not math.isnan(m14) and m14 > f1_max:
        f1_max = m14
        best_method = '0.25s_mks-mcd'
    if not math.isnan(m15) and m15 > f1_max:
        f1_max = m15
        best_method = '1s_mks-mcd'
    if not math.isnan(m16) and m16 > f1_max:
        f1_max = m16
        best_method = '2s_mks-mcd'
    if not math.isnan(m17) and m17 > f1_max:
        f1_max = m17
        best_method = '0.15s_mk-mcd'
    if not math.isnan(m18) and m18 > f1_max:
        f1_max = m18
        best_method = '0.25s_mk-mcd'
    if not math.isnan(m19) and m19 > f1_max:
        f1_max = m19
        best_method = '1s_mk-mcd'
    if not math.isnan(m20) and m20 > f1_max:
        f1_max = m20
        best_method = '2s_mk-mcd'
    if not math.isnan(m21) and m21 > f1_max:
        f1_max = m21
        best_method = '0.15s_ms-mcd'
    if not math.isnan(m22) and m22 > f1_max:
        f1_max = m22
        best_method = '0.25s_ms-mcd'
    if not math.isnan(m23) and m23 > f1_max:
        f1_max = m23
        best_method = '1s_ms-mcd'
    if not math.isnan(m24) and m24 > f1_max:
        f1_max = m24
        best_method = '2s_ms-mcd'
    if not math.isnan(m25) and m25 > f1_max:
        f1_max = m25
        best_method = '0.15s_ks-mcd'
    if not math.isnan(m26) and m26 > f1_max:
        f1_max = m26
        best_method = '0.25s_ks-mcd'
    if not math.isnan(m27) and m27 > f1_max:
        f1_max = m27
        best_method = '1s_ks-mcd'
    if not math.isnan(m28) and m28 > f1_max:
        f1_max = m28
        best_method = '2s_ks-mcd'
    if not math.isnan(m29) and m29 > f1_max:
        f1_max = m29
        best_method = '0.15s_sv-mcd'
    if not math.isnan(m30) and m30 > f1_max:
        f1_max = m30
        best_method = '0.25s_sv-mcd'
    if not math.isnan(m31) and m31 > f1_max:
        f1_max = m31
        best_method = '1s_sv-mcd'
    if not math.isnan(m32) and m32 > f1_max:
        f1_max = m32
        best_method = '2s_sv-mcd'

    print(m_scenario,'=', f1_max, best_method)