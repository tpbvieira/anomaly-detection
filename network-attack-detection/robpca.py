# coding=utf-8
import os
import warnings
import pandas as pd
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")


print('### Contamination: 10%')
label_dir = 'data/ctu_13/raw_clean_test_robpca_csv/10/label/'
result_dir = 'output/ctu_13/results/robpca/10/'
result_file_list = os.listdir(result_dir)
for result_file in result_file_list:
    robpca_result_file = '%s/%s' % (result_dir, result_file)
    robpca_pred = pd.read_csv(robpca_result_file, header=None)
    robpca_pred = robpca_pred[0]
    robpca_pred[robpca_pred == True] = -1
    robpca_pred[robpca_pred == False] = 1
    robpca_pred[robpca_pred == -1] = 0
    robpca_label_file = '%s/label_%s' % (label_dir, result_file)
    ground_truth = pd.read_csv(robpca_label_file, header=None)
    ground_truth = ground_truth[1]
    print(result_file, f1_score(ground_truth, robpca_pred))

print('### Contamination: 25%')
label_dir = 'data/ctu_13/raw_clean_test_robpca_csv/25/label/'
result_dir = 'output/ctu_13/results/robpca/25/'
result_file_list = os.listdir(result_dir)
for result_file in result_file_list:
    robpca_result_file = '%s/%s' % (result_dir, result_file)
    robpca_pred = pd.read_csv(robpca_result_file, header=None)
    robpca_pred = robpca_pred[0]
    robpca_pred[robpca_pred == True] = -1
    robpca_pred[robpca_pred == False] = 1
    robpca_pred[robpca_pred == -1] = 0
    robpca_label_file = '%s/label_%s' % (label_dir, result_file)
    ground_truth = pd.read_csv(robpca_label_file, header=None)
    ground_truth = ground_truth[1]
    print(result_file, f1_score(ground_truth, robpca_pred))

label_dir = 'data/ctu_13/raw_clean_test_robpca_csv/33/label/'
result_dir = 'output/ctu_13/results/robpca/33/'
result_file_list = os.listdir(result_dir)
print('### Contamination: 33%')
for result_file in result_file_list:
    robpca_result_file = '%s/%s' % (result_dir, result_file)
    robpca_pred = pd.read_csv(robpca_result_file, header=None)
    robpca_pred = robpca_pred[0]
    robpca_pred[robpca_pred == True] = -1
    robpca_pred[robpca_pred == False] = 1
    robpca_pred[robpca_pred == -1] = 0
    robpca_label_file = '%s/label_%s' % (label_dir, result_file)
    ground_truth = pd.read_csv(robpca_label_file, header=None)
    ground_truth = ground_truth[1]
    print(result_file, f1_score(ground_truth, robpca_pred))
