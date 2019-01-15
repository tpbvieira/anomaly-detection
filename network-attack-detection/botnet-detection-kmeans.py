# coding=utf-8
import os, gc, time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.cluster import KMeans
from botnet_detection_utils import data_splitting, drop_raw_features, get_classification_report

def getBestByCV(X_train, X_cv, labels):
    # select the best epsilon (threshold) and number of clusters

    # initialize
    best_epsilon = 0
    best_num_clusters = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    for num_clusters in np.arange(1, 10, 1):

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_jobs=-1).fit(X_train)
        X_cv_clusters = kmeans.predict(X_cv)
        X_cv_clusters_centers = kmeans.cluster_centers_

        dist = [np.linalg.norm(x-y) for x,y in zip(X_cv.as_matrix(), X_cv_clusters_centers[X_cv_clusters])]

        y_pred = np.array(dist)

        for epsilon in np.arange(70, 99, 1):
            y_pred[dist >= np.percentile(dist,epsilon)] = 1
            y_pred[dist < np.percentile(dist,epsilon)] = 0

            f1 = f1_score(labels, y_pred, average = "binary")
            Recall = recall_score(labels, y_pred, average = "binary")
            Precision = precision_score(labels, y_pred, average = "binary")

            if f1 > best_f1:
                best_num_clusters = num_clusters
                best_epsilon = epsilon
                best_f1 = f1
                best_precision = Precision
                best_recall = Recall

    return best_num_clusters, best_epsilon, best_f1, best_precision, best_recall

# track execution time
start_time = time.time()

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw_fast/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)
print("## Directory: ", raw_directory)
print("## Files: ", raw_files)

# pickle files have the same names
pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_fast/')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_raw_features.items():
    # initialize labels
    kmeans_test_label = []
    kmeans_pred_test_label = []

    for sample_file in raw_files:
        pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_file_path):
            print("## Sample File: ", pkl_file_path)
            df = pd.read_pickle(pkl_file_path)
        else:
            print("## Sample File: ", raw_file_path)
            raw_df = pd.read_csv(raw_file_path, low_memory=False, dtype={'Label': 'str'})
            df = data_cleasing(raw_df)
        gc.collect()

        # data splitting
        train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_raw_features[features_key])

        # Cross-Validation
        best_num_clusters, best_epsilon, best_f1, best_precision, best_recall = getBestByCV(train_df, cv_df, cv_label)
        print('###[KMeans][',features_key,'] Cross-Validation. Clusters:',best_num_clusters,', Epslilon:',best_epsilon,',F1:',best_f1,', Recall:',best_recall,', Precision:',best_precision)

        # Training - estimate clusters (anomalous or normal) for training
        kmeans = KMeans(n_clusters=best_num_clusters).fit(train_df)

        # Test prediction
        test_clusters = kmeans.predict(test_df)
        test_clusters_centers = kmeans.cluster_centers_

        dist = [np.linalg.norm(x-y) for x, y in zip(test_df.as_matrix(), test_clusters_centers[test_clusters])]

        pred_test_label = np.array(dist)
        pred_test_label[dist >= np.percentile(dist, best_epsilon)] = 1
        pred_test_label[dist < np.percentile(dist, best_epsilon)] = 0

        f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
        print('###[KMeans][', features_key, '] Test. F1:',f1,', Recall:',Recall,', Precision:',Precision)

        kmeans_test_label.extend(test_label.astype(int))  # append into global array
        kmeans_pred_test_label.extend(pred_test_label.astype(int))  # append into global array

    # print results
    f1, Recall, Precision = get_classification_report(kmeans_test_label, kmeans_pred_test_label)
    print('###[KMeans][', features_key, '] Test Full. F1:',f1,', Recall:',Recall,', Precision:',Precision)
print("--- %s seconds ---" % (time.time() - start_time))