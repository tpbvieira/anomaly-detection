# coding=utf-8
import os, warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# result file path
pkl_path = os.path.join('/home/thiago/Downloads/results/21/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
columns = ['F1','method', 'scenarios']
result_df = pd.DataFrame(columns=columns)

# for each file/case
for sample_file in file_list:
    # split the file name into variables
    file_name = sample_file.decode('utf-8')
    tokens = file_name.split('_')
    it = tokens[1]
    scenario = tokens[3]
    method = "%s_%s" % (tokens[0], tokens[2])

    # read the file
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    df = pd.read_pickle(pkl_file_path)
    df = df.T[0]

    # get F1 and variables from the file name
    m_f1 = df.values
    methods = [method] * len(m_f1)
    scenarios = [tokens[3]] * len(m_f1)

    # generate a new dataframe with F1 and variables from the file name
    values_list = list(zip(m_f1, methods, scenarios))
    df = pd.DataFrame(values_list,columns= columns)

    # append into result file
    result_df = result_df.append(df)

m_scenarios = result_df.scenarios.unique()

for m_scenario in m_scenarios:
    fig_name = "%s.png" % (m_scenario)
    result_scenario_df = result_df.loc[result_df['scenarios'] == m_scenario]
    sns.boxplot(x="method", y="F1", data=result_scenario_df, palette="PRGn")
    plt.savefig(fig_name)
    plt.close()
    # break