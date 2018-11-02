import pandas as pd
import os
import gc
from sklearn.metrics import f1_score, recall_score, precision_score


def get_classification_report(m_y_test, m_y_predic):
    m_f1 = f1_score(m_y_test, m_y_predic, average="binary")
    m_recall = recall_score(m_y_test, m_y_predic, average="binary")
    m_precision = precision_score(m_y_test, m_y_predic, average="binary")
    return m_f1, m_recall, m_precision


results_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/mos-eigen-similarity/src/test/matlab/results/')
results_dir = os.fsencode(results_path)
results_dirs = os.listdir(results_dir)

for res_dir in results_dirs:

    res_dir_path = os.path.join(results_dir, res_dir).decode('utf-8')
    res_directory = os.fsencode(res_dir_path)
    res_files = os.listdir(res_directory)

    y_test = pd.read_csv(res_dir_path + '/y_test.csv')

    for res_file in res_files:

        if res_file != 'y_test.csv'.encode('utf-8'):

            res_file_path = os.path.join(res_directory, res_file).decode('utf-8')

            if os.path.isfile(res_file_path):

                y_predict = pd.read_csv(res_file_path)
                f1, recall, precision = get_classification_report(y_test, y_predict)
                print('###[Eigensim][', res_file, '] Test. F1:', f1, ', Recall:', recall, ', Precision:', precision)
                gc.collect()
