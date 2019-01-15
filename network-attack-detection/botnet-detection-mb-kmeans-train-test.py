# coding=utf-8
'''
minibactch-kmeans implementation for train and test files
'''
import pandas as pd
import numpy as np
import os, gc, time
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.cluster import MiniBatchKMeans
from botnet_detection_utils import drop_raw_features, get_classification_report, column_types, data_cleasing


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

            dist = [np.linalg.norm(x-y) for x,y in zip(X_cv.as_matrix(), X_cv_clusters_centers[X_cv_clusters])]

            y_pred = np.array(dist)        

            for m_epsilon in np.arange(70, 95, 2):
                y_pred[dist >= np.percentile(dist,m_epsilon)] = 1
                y_pred[dist < np.percentile(dist,m_epsilon)] = 0
            
                f1 = f1_score(labels, y_pred, average = "binary")
                Recall = recall_score(labels, y_pred, average = "binary")
                Precision = precision_score(labels, y_pred, average = "binary") 

                if f1 > best_f1:
                    best_cluster_size = m_clusters
                    best_batch_size = m_batch_size
                    best_epsilon = m_epsilon
                    best_f1 = f1
                    best_precision = Precision
                    best_recall = Recall

    return best_cluster_size, best_batch_size, best_epsilon, best_f1, best_precision, best_recall


# track execution time
start_time = time.time()

# for each feature set
for features_key, value in drop_raw_features.items():

    # Initialize labels
    mbkmeans_test_label = []
    mbkmeans_pred_test_label = []

    # Load data
    pkl_train_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_all_train2/train.binetflow'
    raw_train_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw_all_train2/train.binetflow'
    pkl_test_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_all_test2/test.binetflow'
    raw_test_file_path = '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw_all_test2/test.binetflow'

    # read pickle or raw dataset for training
    if os.path.isfile(pkl_train_file_path):
        print("## PKL Normal File: ", pkl_train_file_path)
        train_df = pd.read_pickle(pkl_train_file_path)
    else:
        print("## Raw Normal File: ", raw_train_file_path)
        raw_df = pd.read_csv(raw_train_file_path, header = 0, dtype=column_types)
        train_df = data_cleasing(raw_df)
        # save clean data into pickles
        train_df.to_pickle(pkl_train_file_path)
    gc.collect()

    # read pickle or raw dataset for testing
    if os.path.isfile(pkl_test_file_path):
        # print("### Test File: ", pkl_test_file_path)
        test_df = pd.read_pickle(pkl_test_file_path)
    else:
        # print("### Test File: ", raw_test_file_path)
        raw_df = pd.read_csv(raw_test_file_path, header = 0, dtype=column_types)
        test_df = data_cleasing(raw_df)
        # save clean data into pickles
        test_df.to_pickle(pkl_test_file_path)
    gc.collect()
    
    # drop unnecessary features
    train_df.drop(drop_raw_features[features_key], axis =1, inplace = True)
    test_df.drop(drop_raw_features[features_key], axis =1, inplace = True)
    gc.collect()

    # data splitting
    train_len = (len(train_df) * 60) // 100
    cv_df = train_df[train_len+1:]                                      # use the last 40% of training data for cross-validation    
    train_df = train_df[:train_len]                                     # use the first 60% of training data for training
    train_df = train_df[train_df["Label"] == 0]                         # only normal data for training
    train_df = train_df.sort_index()                                    # sort train data
    cv_df = cv_df.sort_index()                                          # sort cv data
    train_df = train_df.drop(labels = ["Label"], axis = 1)              # drop label from training data
    cv_label_df = cv_df["Label"]                                        # save label for testing
    cv_df = cv_df.drop(labels = ["Label"], axis = 1)                    # drop label from testing data
    test_label_df = test_df["Label"]                                    # save label for testing
    test_df = test_df.drop(labels = ["Label"], axis = 1)                # drop label from testing data
    gc.collect()

    # Cross-Validation
    b_clusters, b_batch, b_epsilon, b_f1, b_precision, b_recall = getBestByCV(train_df, cv_df, cv_label_df)
    print('###[MB-KMeans][', features_key, '] Cross-Validation. Clusters:', b_clusters, ', Batch:', b_batch, ', Epslilon:', b_epsilon, ',F1:', b_f1, ', Recall:', b_recall, ', Precision:',
          b_precision)

    # Training - estimate clusters (anomalous or normal) for training    
    mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=b_clusters, batch_size=b_batch, n_init=10, max_no_improvement=10).fit(train_df)
    
    # Test prediction
    test_clusters = mbkmeans.predict(test_df)
    test_clusters_centers = mbkmeans.cluster_centers_
    dist = [np.linalg.norm(x-y) for x,y in zip(test_df.as_matrix(), test_clusters_centers[test_clusters])]
    pred_test_label = np.array(dist)
    pred_test_label[dist >= np.percentile(dist, b_epsilon)] = 1
    pred_test_label[dist < np.percentile(dist, b_epsilon)] = 0

    # print results
    f1, Recall, Precision = get_classification_report(test_label_df.astype(int).values, pred_test_label)
    print('###[MB-KMeans][', features_key, '] Test. F1:', f1, ', Recall:', Recall, ', Precision:', Precision)
print("--- %s seconds ---" % (time.time() - start_time))