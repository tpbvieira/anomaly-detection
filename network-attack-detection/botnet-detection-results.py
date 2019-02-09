# coding=utf-8
import os, warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# result file path
pkl_path = os.path.join('results/pkl_sum/20/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
columns = ['F1', 'methods', 'algs', 'windows', 'scenarios']
result_df = pd.DataFrame(columns=columns)

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
    df = pd.read_pickle(pkl_file_path)
    df = df.T[0]

    # get F1 and variables from the file name
    m_f1 = df.values
    methods = [method] * len(m_f1)
    algs = [alg] * len(m_f1)
    windows = [window] * len(m_f1)
    scenarios = [scenario] * len(m_f1)

    # generate a new dataframe with F1 and variables from the file name
    values_list = list(zip(m_f1, methods, algs, windows, scenarios))
    df = pd.DataFrame(values_list,columns= columns)

    # append into result file
    result_df = result_df.append(df)

m_scenarios = result_df.scenarios.unique()

for m_scenario in m_scenarios:
    # select data
    result_scenario_df = result_df.loc[result_df['scenarios'] == m_scenario]
    result_scenario_df = result_scenario_df.sort_values(['windows', 'algs'], ascending=[True, False])

    # boxplot
    plt.figure(figsize=(15, 5))
    bp = sns.boxplot(x="methods", y="F1", data=result_scenario_df, palette="PRGn", width=0.4)
    fig_name = "results/figures/%s.png" % m_scenario
    plt.savefig(fig_name)
    print(fig_name)
    plt.close()