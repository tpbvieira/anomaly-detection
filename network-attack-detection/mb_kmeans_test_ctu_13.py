# coding=utf-8
import os, gc, time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.cluster import MiniBatchKMeans
from botnet_detection_utils import drop_agg_features, get_classification_report, column_types, data_splitting, data_cleasing


def getBestByCV(X_train, X_cv, labels):
    # select the best epsilon (threshold) and number of clusters

    # initialize
    best_epsilon = 0
    best_cluster_size = 0
    best_batch_size = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    for m_clusters in np.arange(1, 10, 2):

        for m_batch_size in range(10, 100, 10):

            mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=m_clusters, batch_size=m_batch_size, n_init=10, max_no_improvement=10).fit(X_train)

            X_cv_clusters = mbkmeans.predict(X_cv)
            X_cv_clusters_centers = mbkmeans.cluster_centers_

            dist = [np.linalg.norm(x - y) for x, y in zip(X_cv.as_matrix(), X_cv_clusters_centers[X_cv_clusters])]

            y_pred = np.array(dist)

            for m_epsilon in np.arange(70, 95, 2):
                y_pred[dist >= np.percentile(dist, m_epsilon)] = 1
                y_pred[dist < np.percentile(dist, m_epsilon)] = 0

                f1 = f1_score(labels, y_pred, average="binary")
                Recall = recall_score(labels, y_pred, average="binary")
                Precision = precision_score(labels, y_pred, average="binary")

                if f1 > best_f1:
                    best_cluster_size = m_clusters
                    best_batch_size = m_batch_size
                    best_epsilon = m_epsilon
                    best_f1 = f1
                    best_precision = Precision
                    best_recall = Recall

    return best_cluster_size, best_batch_size, best_epsilon, best_f1, best_precision, best_recall


start_time = time.time()

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum_fast/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_sum_fast/')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_agg_features.items():

    # Initialize labels
    mbkmeans_test_label = []
    mbkmeans_pred_test_label = []

    for sample_file in raw_files:

        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')

        # read pickle file with pandas or...
        if os.path.isfile(pkl_file_path):
            print("## Sample File: ", pkl_file_path)
            df = pd.read_pickle(pkl_file_path)
        else:  # load raw file and save clean data into pickles
            print("## Sample File: ", raw_file_path)
            raw_df = pd.read_csv(raw_file_path, header=0, dtype=column_types)
            df = data_cleasing(raw_df)
            df.to_pickle(pkl_file_path)
        gc.collect()

        # data splitting
        norm_train_df, cv_df, test_df, cv_label_df, test_label_df = data_splitting(df, drop_agg_features[features_key])

        # Cross-Validation
        b_clusters, b_batch, b_epsilon, b_f1, b_precision, b_recall = getBestByCV(norm_train_df, cv_df, cv_label_df)
        print('###[MB-KMeans][', features_key, '] Cross-Validation. Clusters:',b_clusters,', Batch:',b_batch,', Epslilon:',b_epsilon,',F1:',b_f1,', Recall:',b_recall,', Precision:',b_precision)

        # Training - estimate clusters (anomalous or normal) for training
        mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=b_clusters, batch_size=b_batch, n_init=10, max_no_improvement=10).fit(norm_train_df)

        # Test prediction
        test_clusters = mbkmeans.predict(test_df)
        test_clusters_centers = mbkmeans.cluster_centers_
        dist = [np.linalg.norm(x - y) for x, y in zip(test_df.as_matrix(), test_clusters_centers[test_clusters])]
        pred_test_label = np.array(dist)
        pred_test_label[dist >= np.percentile(dist, b_epsilon)] = 1
        pred_test_label[dist < np.percentile(dist, b_epsilon)] = 0
        test_label = test_label_df.astype(int).values

        # print results
        f1, Recall, Precision = get_classification_report(test_label, pred_test_label)
        print('###[MB-KMeans][', features_key, '] Test. F1:',f1,', Recall:',Recall,', Precision:',Precision)

        # save results for total evaluation later
        mbkmeans_test_label.extend(test_label)
        mbkmeans_pred_test_label.extend(pred_test_label)

    f1, Recall, Precision = get_classification_report(mbkmeans_test_label, mbkmeans_pred_test_label)
    print('###[MB-KMeans][', features_key, '] Test Full. F1:',f1,', Recall:',Recall,', Precision:',Precision)
print("--- %s seconds ---" % (time.time() - start_time))