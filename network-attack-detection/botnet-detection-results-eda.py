# coding=utf-8
import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")


def get_outliers(m_dist, m_best_contamination):
    m_pred_label = np.full(m_dist.shape[0], 0, dtype=int)
    m_best_contamination = np.percentile(m_dist, 100. * m_best_contamination)
    m_pred_label[m_dist <= m_best_contamination] = 1
    return m_pred_label

# init variables
it = 20

# result file path
pkl_path = os.path.join('results/pkl_sum_dict/%s/data/' % it)
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
result_columns = ['f1', 'approaches', 'algs', 'windows', 'scenarios']
result_df = pd.DataFrame(columns=result_columns)

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
    test_label_series = result_dict.get(0).get('test_label')
    best_cont = result_dict.get(0).get('train_best_cont')
    best_train_f1 = result_dict.get(0).get('train_best_f1')
    counts = test_label_series.value_counts()
    total_counts =  len(test_label_series)
    print('###[botnet-detection-results-eda] ', sample_file)
    print("\tTotal:", total_counts, ", 0:", counts.get(0),'(', round((counts.get(0)/total_counts)*100, 2),'%), 1:',
          counts.get(1),'(', round((counts.get(1)/total_counts)*100, 2),'%), best_contamination:', best_cont,
          ', best_train_f1:', best_train_f1)

    # for each iteration compute F1 and save the algorithm, window size, scenario and approach (window size and algorithm))
    for key, value in result_dict.items():

        # get saved contamination, labels and distances
        best_contamination = result_dict.get(key).get('train_best_cont')
        mcd_pred_dist_ = result_dict.get(key).get('test_mcd_dist_')
        mcd2_pred_dist_ = result_dict.get(key).get('test_mcd2_dist_')
        k_pred_dist_ = result_dict.get(key).get('test_k_dist_')
        s_pred_dist_ = result_dict.get(key).get('test_s_dist_')
        m_label = result_dict.get(key).get('test_mcd_label')
        m2_label = result_dict.get(key).get('test_mcd2_label')
        k_label = result_dict.get(key).get('test_k_label')
        s_label = result_dict.get(key).get('test_s_label')
        test_label = result_dict.get(key).get('test_label')

        # combine distances
        mks_dist_ = mcd_pred_dist_ - k_pred_dist_ - s_pred_dist_
        mk_dist_ = mcd_pred_dist_ - k_pred_dist_
        ms_dist_ = mcd_pred_dist_ - s_pred_dist_
        m2ks_dist_ = mcd2_pred_dist_ - k_pred_dist_ - s_pred_dist_
        m2k_dist_ = mcd2_pred_dist_ - k_pred_dist_
        m2s_dist_ = mcd2_pred_dist_ - s_pred_dist_
        ks_dist_ = k_pred_dist_ + s_pred_dist_

        # save distances and label into a dataframe
        dists_columns = ['mcd_dist', 'mcd2_dist', 'k_dist', 's_dist', 'mks_dist', 'mk_dist', 'ms_dist', 'ks_dist', 'Label']
        dists_df = pd.DataFrame(columns=dists_columns)
        dists_df['mcd_dist'] = mcd_pred_dist_
        dists_df['mcd2_dist'] = mcd2_pred_dist_
        dists_df['k_dist'] = k_pred_dist_
        dists_df['s_dist'] = s_pred_dist_
        dists_df['mks_dist'] = mks_dist_
        dists_df['mk_dist'] = mk_dist_
        dists_df['ms_dist'] = ms_dist_
        dists_df['ks_dist'] = ks_dist_.copy()
        dists_df['Label'] =  test_label.values

        # # get outliers from new distances, however the labels should be the same of saved labels
        # k_label = get_outliers(k_pred_dist_, best_contamination)
        # s_label = get_outliers(s_pred_dist_, best_contamination)

        # # ensemble with labels
        # mks_label = m_label.copy()
        # mks_label.fill(0)
        # mks_label[np.logical_or(m_label.copy(), k_label.copy(), s_label.copy())] = 1
        # mk_label = m_label.copy()
        # mk_label.fill(0)
        # mk_label[np.logical_or(m_label.copy(), k_label).copy()] = 1
        # ms_label = m_label.copy()
        # ms_label.fill(0)
        # ms_label[np.logical_or(m_label.copy(), s_label.copy())] = 1
        # ks_label = m_label.copy()
        # ks_label.fill(0)
        # ks_label[np.logical_or(k_label.copy(), s_label.copy())] = 1

        # ensemble with distances
        mks_label = get_outliers(mks_dist_, best_contamination)
        mk_label = get_outliers(mk_dist_, best_contamination)
        ms_label = get_outliers(ms_dist_, best_contamination)
        m2ks_label = get_outliers(m2ks_dist_, best_contamination)
        m2k_label = get_outliers(m2k_dist_, best_contamination)
        m2s_label = get_outliers(m2s_dist_, best_contamination)
        ks_label = get_outliers(ks_dist_, best_contamination)

        # compute F1 score and append results into a mcd dataframe
        m_pred_f1 = f1_score(m_label, test_label, average="binary")
        alg = 'mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m_pred_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a mcd2 dataframe
        m_pred_f1 = f1_score(m2_label, test_label, average="binary")
        alg = 'mcd2'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m_pred_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a k-mcd dataframe
        k_pred_f1 = f1_score(k_label, test_label, average="binary")
        alg = 'k-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[k_pred_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a s-mcd dataframe
        s_pred_f1 = f1_score(s_label, test_label, average="binary")
        alg = 's-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[s_pred_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a mks-mcd dataframe
        mks_f1 = f1_score(mks_label, test_label, average="binary")
        alg = 'mks-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[mks_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a mk-mcd dataframe
        mk_f1 = f1_score(mk_label, test_label, average="binary")
        alg = 'mk-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[mk_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a ms-mcd dataframe
        ms_f1 = f1_score(ms_label, test_label, average="binary")
        alg = 'ms-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[ms_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a m2ks-mcd dataframe
        m2ks_f1 = f1_score(m2ks_label, test_label, average="binary")
        alg = 'm2ks-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m2ks_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a m2k-mcd dataframe
        m2k_f1 = f1_score(m2k_label, test_label, average="binary")
        alg = 'm2k-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m2k_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a m2s-mcd dataframe
        m2s_f1 = f1_score(m2s_label, test_label, average="binary")
        alg = 'm2s-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m2s_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # compute F1 score and append results into a ks-mcd dataframe
        ks_f1 = f1_score(ks_label, test_label, average="binary")
        alg = 'ks-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[ks_f1, approach, alg, window, scenario]], columns=result_columns)
        result_df = result_df.append(df)

        # only for the first iteration, plot distances into boxplot and pairplot
        if key == 0:
            m_columns = ['mcd_dist_', 'mcd2_dist_', 'k_dist_', 's_dist_', 'mks_dist_', 'mk_dist_', 'ms_dist_',
                     'm2ks_dist_', 'm2k_dist_', 'm2s_dist_', 'ks_dist_', 'test_label']
            result_scenario_df = pd.DataFrame(list(zip(mcd_pred_dist_, mcd2_pred_dist_, k_pred_dist_, s_pred_dist_,
                                                       mks_dist_, mk_dist_, ms_dist_, m2ks_dist_, m2k_dist_, m2s_dist_,
                                                       ks_dist_, test_label)), columns=m_columns)

            # boxplot mcd_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_mcd_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="mcd_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t',fig_name)

            # boxplot mcd2_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_mcd2_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="mcd2_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot k_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_k_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="k_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t',fig_name)

            # boxplot s_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_s_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="s_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t',fig_name)

            # boxplot mks_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_mks_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="mks_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot mk_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_mk_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="mk_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot ms_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_ms_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="ms_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot mks_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_m2ks_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="m2ks_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot mk_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_m2k_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="m2k_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot ms_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_m2s_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="m2s_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # boxplot ks_dist_
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_ks_dist.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plt.figure(figsize=(15, 5))
                bp = sns.boxplot(x="test_label", y="ks_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
                bp.set(yscale="log")
                plt.savefig(fig_name)
                plt.close()
                print('\t', fig_name)

            # pairplot 'mcd_dist_', 'k_dist_', 's_dist_'
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_distances.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plot_features = ['mcd_dist_', 'k_dist_', 's_dist_']
                sns_distplot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label', diag_kind='kde',
                                            plot_kws={'alpha': 0.1, 's': 80, 'edgecolor': 'k'}, size=4)
                sns_distplot.savefig(fig_name)
                print('\t', fig_name)

            # pairplot 'mcd2_dist_', 'k_dist_', 's_dist_'
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_distances2.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plot_features = ['mcd2_dist_', 'k_dist_', 's_dist_']
                sns_distplot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label',
                                            diag_kind='kde',
                                            plot_kws={'alpha': 0.1, 's': 80, 'edgecolor': 'k'}, size=4)
                sns_distplot.savefig(fig_name)
                print('\t', fig_name)

            # pairplot of ensemble 'mks_dist_', 'mk_dist_', 'ms_dist_', 'ks_dist_'
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_distances_new.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plot_features = ['mks_dist_', 'mk_dist_', 'ms_dist_', 'ks_dist_']
                sns_distplot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label',
                                            diag_kind='kde',
                                            plot_kws={'alpha': 0.1, 's': 80, 'edgecolor': 'k'}, size=4)
                sns_distplot.savefig(fig_name)
                print('\t', fig_name)

            # pairplot of ensemble 'm2ks_dist_', 'm2k_dist_', 'm2s_dist_', 'ks_dist_'
            fig_name = "results/pkl_sum_dict/%s/figures/%s_%s_distances_new2.png" % (it, scenario, window)
            if not os.path.isfile(fig_name):
                plot_features = ['m2ks_dist_', 'm2k_dist_', 'm2s_dist_', 'ks_dist_']
                sns_distplot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label',
                                            diag_kind='kde',
                                            plot_kws={'alpha': 0.1, 's': 80, 'edgecolor': 'k'}, size=4)
                sns_distplot.savefig(fig_name)
                print('\t', fig_name)

print('\n### Highest median per scenario')
# for each scenario (original file name, such as 10 or 18-2)
m_scenarios = result_df.scenarios.unique()
for m_scenario in m_scenarios:

    # get data by scenario and group by window and algorithm
    result_scenario_df = result_df.loc[result_df['scenarios'] == m_scenario]
    result_scenario_df = result_scenario_df.sort_values(['windows', 'algs'], ascending=[True, False])

    # boxplot F1 score by approaches for this scenario
    fig_name = "results/pkl_sum_dict/%s/figures/%s.png" % (it, m_scenario)
    if not os.path.isfile(fig_name):
        plt.figure(figsize=(15, 5))
        bp = sns.boxplot(x="approaches", y="f1", data=result_scenario_df, palette="PRGn", width=0.4)
        bp.set_xticklabels(bp.get_xticklabels(),rotation=90)
        plt.subplots_adjust(bottom=0.15)
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
    m29 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_mcd2'].median().get('f1')
    m30 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_mcd2'].median().get('f1')
    m31 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_mcd2'].median().get('f1')
    m32 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_mcd2'].median().get('f1')
    m33 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_m2ks-mcd'].median().get('f1')
    m34 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_m2ks-mcd'].median().get('f1')
    m35 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_m2ks-mcd'].median().get('f1')
    m36 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_m2ks-mcd'].median().get('f1')
    m37 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_m2k-mcd'].median().get('f1')
    m38 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_m2k-mcd'].median().get('f1')
    m39 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_m2k-mcd'].median().get('f1')
    m40 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_m2k-mcd'].median().get('f1')
    m41 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.15s_m2s-mcd'].median().get('f1')
    m42 = result_scenario_df.loc[result_scenario_df['approaches'] == '0.25s_m2s-mcd'].median().get('f1')
    m43 = result_scenario_df.loc[result_scenario_df['approaches'] == '1s_m2s-mcd'].median().get('f1')
    m44 = result_scenario_df.loc[result_scenario_df['approaches'] == '2s_m2s-mcd'].median().get('f1')

    # detect the approach with the largest F1 score for the current scenario and print it
    hi_median_f1 = 0
    se_median_f1 = 0
    best_method = ''
    se_best_method = ''
    if not math.isnan(m1) and m1 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m1
        best_method = '0.15s_s-mcd'
    if not math.isnan(m2) and m2 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m2
        best_method = '0.25s_s-mcd'
    if not math.isnan(m3) and m3 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m3
        best_method = '1s_s-mcd'
    if not math.isnan(m4) and m4 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m4
        best_method = '2s_s-mcd'
    if not math.isnan(m5) and m5 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m5
        best_method = '0.15s_k-mcd'
    if not math.isnan(m6) and m6 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m6
        best_method = '0.25s_k-mcd'
    if not math.isnan(m7) and m7 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m7
        best_method = '1s_k-mcd'
    if not math.isnan(m8) and m8 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m8
        best_method = '2s_k-mcd'
    # if not math.isnan(m9) and m9 > hi_median_f1:
    #     hi_median_f1 = m9
    #     best_method = '0.15s_mcd'
    # if not math.isnan(m10) and m10 > hi_median_f1:
    #     hi_median_f1 = m10
    #     best_method = '0.25s_mcd'
    # if not math.isnan(m11) and m11 > hi_median_f1:
    #     hi_median_f1 = m11
    #     best_method = '1s_mcd'
    # if not math.isnan(m12) and m12 > hi_median_f1:
    #     hi_median_f1 = m12
    #     best_method = '2s_mcd'
    if not math.isnan(m13) and m13 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m13
        best_method = '0.15s_mks-mcd'
    if not math.isnan(m14) and m14 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m14
        best_method = '0.25s_mks-mcd'
    if not math.isnan(m15) and m15 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m15
        best_method = '1s_mks-mcd'
    if not math.isnan(m16) and m16 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m16
        best_method = '2s_mks-mcd'
    if not math.isnan(m17) and m17 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m17
        best_method = '0.15s_mk-mcd'
    if not math.isnan(m18) and m18 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m18
        best_method = '0.25s_mk-mcd'
    if not math.isnan(m19) and m19 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m19
        best_method = '1s_mk-mcd'
    if not math.isnan(m20) and m20 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m20
        best_method = '2s_mk-mcd'
    if not math.isnan(m21) and m21 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m21
        best_method = '0.15s_ms-mcd'
    if not math.isnan(m22) and m22 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m22
        best_method = '0.25s_ms-mcd'
    if not math.isnan(m23) and m23 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m23
        best_method = '1s_ms-mcd'
    if not math.isnan(m24) and m24 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m24
        best_method = '2s_ms-mcd'
    if not math.isnan(m25) and m25 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m25
        best_method = '0.15s_ks-mcd'
    if not math.isnan(m26) and m26 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m26
        best_method = '0.25s_ks-mcd'
    if not math.isnan(m27) and m27 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m27
        best_method = '1s_ks-mcd'
    if not math.isnan(m28) and m28 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m28
        best_method = '2s_ks-mcd'
    if not math.isnan(m29) and m29 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m29
        best_method = '0.15s_mcd2'
    if not math.isnan(m30) and m30 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m30
        best_method = '0.25s_mcd2'
    if not math.isnan(m31) and m31 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m31
        best_method = '1s_mcd2'
    if not math.isnan(m32) and m32 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m32
        best_method = '2s_mcd2'
    if not math.isnan(m33) and m33 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m33
        best_method = '0.15s_m2ks-mcd'
    if not math.isnan(m34) and m34 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m34
        best_method = '0.25s_m2ks-mcd'
    if not math.isnan(m35) and m35 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m35
        best_method = '1s_m2ks-mcd'
    if not math.isnan(m36) and m36 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m36
        best_method = '2s_m2ks-mcd'
    if not math.isnan(m37) and m37 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m37
        best_method = '0.15s_m2k-mcd'
    if not math.isnan(m38) and m38 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m38
        best_method = '0.25s_m2k-mcd'
    if not math.isnan(m39) and m39 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m39
        best_method = '1s_m2k-mcd'
    if not math.isnan(m40) and m40 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m40
        best_method = '2s_m2k-mcd'
    if not math.isnan(m41) and m41 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m41
        best_method = '0.15s_m2s-mcd'
    if not math.isnan(m42) and m42 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m42
        best_method = '0.25s_m2s-mcd'
    if not math.isnan(m43) and m43 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m43
        best_method = '1s_m2s-mcd'
    if not math.isnan(m44) and m44 > hi_median_f1:
        se_median_f1 = hi_median_f1
        se_best_method = best_method
        hi_median_f1 = m44
        best_method = '2s_m2s-mcd'
    scenario_f1_alg_str = "\t%s = %f/%s\t%f/%s" % (m_scenario, hi_median_f1, best_method, se_median_f1, se_best_method)
    print(scenario_f1_alg_str)

print('\n### Global median F1 per algorithm and window size')
# for each algorithm, group by window (k-mcd_1s, for example), and print the median of global F1 score
m_algs = result_df.algs.unique()
for m_alg in m_algs:
    result_alg_df = result_df.loc[result_df['algs'] == m_alg]

    m_windows = result_alg_df.windows.unique()
    for m_window in m_windows:
        alg_window_f1 = result_alg_df.loc[result_alg_df['windows'] == m_window].median().get('f1')
        alg_window_f1_str = "\t%s_%s_%f" % (m_alg, m_window, alg_window_f1)
        print(alg_window_f1_str)

# boxplot global F1 grouped by approaches (window_algorithm)
fig_name = "results/pkl_sum_dict/%s/figures/all_f1_boxplot.png" % it
if not os.path.isfile(fig_name):
    plt.figure(figsize=(25, 5))
    bp = sns.boxplot(x="approaches", y="f1", data=result_df, palette="PRGn", width=0.6)
    bp.set_xticklabels(bp.get_xticklabels(),rotation=90)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(fig_name)
    plt.close()