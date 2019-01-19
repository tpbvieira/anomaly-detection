# coding=utf-8
import os, warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# result file path
pkl_path = os.path.join('/home/thiago/Downloads/mcd/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# initialize the dataframe
columns = ['F1','Alg', 'windows', 'scenarios']
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

    # read the file
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    df = pd.read_pickle(pkl_file_path)
    df = df.T[0]
    df = df.loc[:3]

    # get F1 and variables from the file name
    m_f1 = df.values
    algs = [tokens[0]] * len(m_f1)
    windows = [tokens[2]] * len(m_f1)
    scenarios = [tokens[3]] * len(m_f1)

    # generate a new dataframe with F1 and variables from the file name
    values_list = list(zip(m_f1, algs, windows, scenarios))
    df = pd.DataFrame(values_list,columns= columns)

    # append into result file
    result_df = result_df.append(df)

# plot grouped boxplot
sns.boxplot(x="scenarios", y="F1", hue="windows", data=result_df, palette="PRGn")
plt.show()