# coding=utf-8
import os, warnings
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
warnings.filterwarnings("ignore")


def get_classification_report(m_y_test, m_y_predic):
    m_f1 = f1_score(m_y_test, m_y_predic, average="binary")
    m_recall = recall_score(m_y_test, m_y_predic, average="binary")
    m_precision = precision_score(m_y_test, m_y_predic, average="binary")
    return m_f1, m_recall, m_precision


results_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/eigensim/src/test/matlab/results/')
results_dir = os.fsencode(results_path)
results_dirs = os.listdir(results_dir)

# for each subdir
for res_dir in results_dirs:
    print('### ', res_dir)
    res_dir_path = os.path.join(results_dir, res_dir).decode('utf-8')
    if os.path.isdir(res_dir_path):

        res_directory = os.fsencode(res_dir_path)
        res_files = os.listdir(res_directory)

        # read test file for comparison
        y_test = pd.read_csv(res_dir_path + '/y_test.csv')

        # for each file in subdir
        for res_file in res_files:

            if res_file != 'y_test.csv'.encode('utf-8'):

                res_file_path = os.path.join(res_directory, res_file).decode('utf-8')
                if os.path.isfile(res_file_path):
                    y_predict = pd.read_csv(res_file_path)
                    try:
                        f1, recall, precision = get_classification_report(y_test, y_predict)
                        print('###[Eigensim][', res_file, '] Test. F1:', f1, ', Recall:', recall, ', Precision:', precision)
                    except:
                        a = 1
                        # print(">>> Unexpected error:", res_file_path)