import pandas as pd
from sklearn.metrics import classification_report, average_precision_score, f1_score, recall_score, precision_score

y_test = pd.read_csv('/home/thiago/dev/projects/discriminative-sensing/mos-eigen-similarity/src/test/matlab/train_label.csv', header=None)
print(y_test.shape)
y_predic = pd.read_csv('/home/thiago/dev/projects/discriminative-sensing/mos-eigen-similarity/src/test/matlab/20_6_unit_eig_akaike_0.1.csv')
print(y_predic.shape)

f = f1_score(y_test, y_predic, average = "binary")
print("### F1Score: ", f)