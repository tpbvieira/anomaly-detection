# coding=utf-8
import os, warnings, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# result file path
pkl_path = os.path.join('results/pkl_sum_dict/20/data/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
columns = ['mcd_prediction_dist_', 'k_prediction_dist_', 's_prediction_dist_', 'test_label', 'methods', 'algs', 'windows', 'scenarios']
result_df = pd.DataFrame(columns=columns)

# "mcd_test_label": mcd_pred_test_label,
# "k_test_label": k_pred_test_label,
# "s_test_label": s_pred_test_label,

# for each file/case
for sample_file in file_list:
    # split the file name into variables
    file_name = sample_file.decode('utf-8')
    tokens = file_name.split('_')
    alg = tokens[0]
    it = tokens[1]
    window = tokens[2]
    scenario = tokens[3]
    method = "%s_%s" % (alg, window)

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
    print('\n###[botnet-detection-results-eda] ', sample_file)
    print("Total:", total_counts, ", 0:", counts.get(0),'(', round((counts.get(0)/total_counts)*100, 2),'%), 1:',
          counts.get(1),'(', round((counts.get(1)/total_counts)*100, 2),'%), best_contamination:', best_cont,
          ', best_train_f1:', best_train_f1)

    scenario_df = pd.DataFrame(columns=columns)
    for key, value in result_dict.items():

        mcd_prediction_dist_ = result_dict.get(key).get('mcd_prediction_dist_')
        k_prediction_dist_ = result_dict.get(key).get('k_prediction_dist_')
        s_prediction_dist_ = result_dict.get(key).get('s_prediction_dist_')
        test_label = result_dict.get(key).get('test_label')
        methods = [method] * len(test_label)
        algs = [alg] * len(test_label)
        windows = [window] * len(test_label)
        scenarios = [scenario] * len(test_label)

        # generate a new dataframe with F1 and variables from the file name
        values_list = list(zip(mcd_prediction_dist_, k_prediction_dist_, s_prediction_dist_, test_label, methods, algs, windows, scenarios))
        df = pd.DataFrame(values_list,columns= columns)

        # append into result file
        result_df = result_df.append(df)
        continue

# # get possible scenarios
# m_scenarios = result_df.scenarios.unique()
#
# for m_scenario in m_scenarios:
#     # select data
#     result_scenario_df = result_df.loc[result_df['scenarios'] == m_scenario]
#     result_scenario_df = result_scenario_df.sort_values(['windows', 'algs'], ascending=[True, False])
#
#     # boxplot prediction_dist_
#     plt.figure(figsize=(15, 5))
#     bp = sns.boxplot(x="test_label", y="mcd_prediction_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
#     # bp.set(yscale="log")
#     fig_name = "results/pkl_sum_dict/20/figures/mcd_prediction_dist_%s.png" % m_scenario
#     plt.savefig(fig_name)
#     plt.close()
#     print(fig_name)
#
#     # boxplot prediction_dist_
#     plt.figure(figsize=(15, 5))
#     bp = sns.boxplot(x="test_label", y="k_prediction_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
#     # bp.set(yscale="log")
#     fig_name = "results/pkl_sum_dict/20/figures/k_prediction_dist_%s.png" % m_scenario
#     plt.savefig(fig_name)
#     plt.close()
#     print(fig_name)
#
#     # boxplot prediction_dist_
#     plt.figure(figsize=(15, 5))
#     bp = sns.boxplot(x="test_label", y="s_prediction_dist_", data=result_scenario_df, palette="PRGn", width=0.4)
#     # bp.set(yscale="log")
#     fig_name = "results/pkl_sum_dict/20/figures/s_prediction_dist_%s.png" % m_scenario
#     plt.savefig(fig_name)
#     plt.close()
#     print(fig_name)
#
#     plot_features = ['mcd_prediction_dist_', 'k_prediction_dist_', 's_prediction_dist_']
#     sns_plot = sns.pairplot(result_scenario_df, vars=plot_features, hue='test_label')
#     fig_name = "results/pkl_sum_dict/20/figures/distances_%s_pairplot.png" % m_scenario
#     sns_plot.savefig(fig_name)
#     print(fig_name)