# coding=utf-8
import os, warnings, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

# result file path
pkl_path = os.path.join('results/pkl_sum_dict/20/data/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
columns = ['f1', 'approaches', 'algs', 'windows', 'scenarios']
result_df = pd.DataFrame(columns=columns)

# for each scenario (file name)
for sample_file in file_list:
    # split the file name into variables
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

        # get data
        mcd_pred_dist_ = result_dict.get(key).get('mcd_prediction_dist_')
        k_pred_dist_ = result_dict.get(key).get('k_prediction_dist_')
        s_pred_dist_ = result_dict.get(key).get('s_prediction_dist_')
        mcd_pred_label = result_dict.get(key).get('mcd_test_label')
        k_pred_label = result_dict.get(key).get('k_test_label')
        s_pred_label = result_dict.get(key).get('s_test_label')
        test_label = result_dict.get(key).get('test_label')

        # compute F1 score
        m_f1 = f1_score(mcd_pred_label, test_label, average="binary")
        k_f1 = f1_score(k_pred_label, test_label, average="binary")
        s_f1 = f1_score(s_pred_label, test_label, average="binary")

        # generate a new dataframe with F1 and variables from the file name
        alg = 'mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[m_f1, approach, alg, window, scenario]],columns= columns)
        result_df = result_df.append(df)
        alg = 'k-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[k_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)
        alg = 's-mcd'
        approach = "%s_%s" % (window, alg)
        df = pd.DataFrame([[s_f1, approach, alg, window, scenario]], columns=columns)
        result_df = result_df.append(df)

        if key == 0:
            m_columns = ['mcd_prediction_dist_', 'k_prediction_dist_', 's_prediction_dist_', 'test_label']
            result_scenario_df = pd.DataFrame(list(zip(mcd_pred_dist_, k_pred_dist_, s_pred_dist_, test_label)),
                                              columns=m_columns)

            # boxplot prediction_dist_
            plt.figure(figsize=(15, 5))
            bp = sns.boxplot(x="test_label", y="mcd_prediction_dist_", data=result_scenario_df, palette="PRGn",
                             width=0.4)
            # bp.set(yscale="log")
            fig_name = "results/pkl_sum_dict/20/figures/mcd_prediction_dist_%s_%s.png" % (scenario, approach)
            plt.savefig(fig_name)
            plt.close()
            print('\t',fig_name)

            # boxplot prediction_dist_
            plt.figure(figsize=(15, 5))
            bp = sns.boxplot(x="test_label", y="k_prediction_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
            # bp.set(yscale="log")
            fig_name = "results/pkl_sum_dict/20/figures/k_prediction_dist_%s_%s.png" % (scenario, approach)
            plt.savefig(fig_name)
            plt.close()
            print('\t',fig_name)

            # boxplot prediction_dist_
            plt.figure(figsize=(15, 5))
            bp = sns.boxplot(x="test_label", y="s_prediction_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
            # bp.set(yscale="log")
            fig_name = "results/pkl_sum_dict/20/figures/s_prediction_dist_%s_%s.png" % (scenario, approach)
            plt.savefig(fig_name)
            plt.close()
            print('\t',fig_name)

            plot_features = ['mcd_prediction_dist_', 'k_prediction_dist_', 's_prediction_dist_']
            sns_plot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label', diag_kind = 'kde',
             plot_kws = {'alpha': 0.1, 's': 80, 'edgecolor': 'k'},
             size = 4);
            fig_name = "results/pkl_sum_dict/20/figures/distances_%s_%s_pairplot.png" % (scenario, approach)
            sns_plot.savefig(fig_name)
            print('\t',fig_name)

m_scenarios = result_df.scenarios.unique()
for m_scenario in m_scenarios:
    # select data
    result_scenario_df = result_df.loc[result_df['scenarios'] == m_scenario]
    result_scenario_df = result_scenario_df.sort_values(['windows', 'algs'], ascending=[True, False])

    # boxplot
    plt.figure(figsize=(15, 5))
    bp = sns.boxplot(x="approahes", y="f1", data=result_scenario_df, palette="PRGn", width=0.4)
    fig_name = "results/pkl_sum_dict/20/figures/%s.png" % m_scenario
    plt.savefig(fig_name)
    print('\t',fig_name)
    plt.close()

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

    f1_max = m1
    best_method = '0.15s_s-mcd'
    if m2 > f1_max:
        f1_max = m2
        best_method = '0.25s_s-mcd'
    if m3 > f1_max:
        f1_max = m3
        best_method = '1s_s-mcd'
    if m4 > f1_max:
        f1_max = m4
        best_method = '2ss-mcd'
    if m5 > f1_max:
        f1_max = m5
        best_method = '0.15s_k-mcd'
    if m6 > f1_max:
        f1_max = m6
        best_method = 'k-mcd_0.25s'
    if m7 > f1_max:
        f1_max = m7
        best_method = '1s_k-mcd'
    if m8 > f1_max:
        f1_max = m8
        best_method = '2s_k-mcd'
    if m9 > f1_max:
        f1_max = m9
        best_method = '0.15s_mcd'
    if m10 > f1_max:
        f1_max = m10
        best_method = '0.25s_mcd'
    if m11 > f1_max:
        f1_max = m11
        best_method = '1s_mcd'
    if m12 > f1_max:
        f1_max = m12
        best_method = '2s_mcd'

    print(m_scenario,'=', f1_max, best_method)