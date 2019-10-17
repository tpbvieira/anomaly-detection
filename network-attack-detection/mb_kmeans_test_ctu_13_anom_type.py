# coding=utf-8
'''
minibactch-kmeans implementation for normal vs attack or normal vs CC data
'''
import pandas as pd
import numpy as np
import os, gc, time
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.cluster import MiniBatchKMeans
from botnet_detection_utils import drop_raw_features, get_classification_report, data_cleasing, column_types


def data_merge_splitting(normal_df, anom_df):

    # Length and indexes
    norm_len = len(normal_df.index)
    norm_train_end = (norm_len * 60) // 100
    norm_cv_start = norm_train_end + 1
    norm_cv_end = (norm_len * 80) // 100
    norm_test_start = norm_cv_end + 1
    anom_len = len(anom_df.index)
    anom_cv_end = anom_len // 2
    anom_test_start = anom_cv_end + 1

    # split normal data
    norm_train_df = normal_df[:norm_train_end]
    norm_cv_df = normal_df[norm_cv_start:norm_cv_end]
    norm_test_df = normal_df[norm_test_start:norm_len]
    gc.collect()
    
    # split anomalous data
    anom_cv_df  = anom_df[:anom_cv_end]
    anom_test_df = anom_df[anom_test_start:anom_len]
    gc.collect()

    # CV and test data from concatenation of normal and anomalous data
    cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
    test_df = pd.concat([norm_test_df, anom_test_df], axis=0)
    gc.collect()

    # Sort data by index
    norm_train_df = norm_train_df.sort_index()
    cv_df = cv_df.sort_index()
    test_df = test_df.sort_index()
    gc.collect()

    # save labels and drop labels from data
    cv_label_df = cv_df["Label"]
    test_label_df = test_df["Label"]
    norm_train_df = norm_train_df.drop(labels = ["Label"], axis = 1)
    cv_df = cv_df.drop(labels = ["Label"], axis = 1)
    test_df = test_df.drop(labels = ["Label"], axis = 1)

    gc.collect()
    
    return norm_train_df, cv_df, test_df, cv_label_df, test_label_df


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

raw_normal_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw_normal/')
raw_normal_directory = os.fsencode(raw_normal_path)
raw_normal_files = os.listdir(raw_normal_directory)
raw_anom_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw_attack/')
raw_anom_directory = os.fsencode(raw_anom_path)
raw_anom_files = os.listdir(raw_anom_directory)

pkl_normal_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_normal/')
pkl_normal_directory = os.fsencode(pkl_normal_path)
pkl_anom_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl_attack/')
pkl_anom_directory = os.fsencode(pkl_anom_path)

# for each feature set
for features_key, value in drop_raw_features.items():

    all_normal_df = None
    all_anom_df = None

    for sample_file in raw_normal_files:
        
        raw_normal_file_path = os.path.join(raw_normal_directory, sample_file).decode('utf-8')    
        raw_anom_file_path = os.path.join(raw_anom_directory, sample_file).decode('utf-8')

        pkl_normal_file_path = os.path.join(pkl_normal_directory, sample_file).decode('utf-8')
        pkl_anom_file_path = os.path.join(pkl_anom_directory, sample_file).decode('utf-8')        

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_normal_file_path):
            print("## PKL Normal File: ", pkl_normal_file_path)
            normal_df = pd.read_pickle(pkl_normal_file_path)
        else:
            print("## Raw Normal File: ", raw_normal_file_path)
            normal_df = pd.read_csv(raw_normal_file_path, low_memory=True, header=0, dtype=column_types)
            normal_df = data_cleasing(normal_df)
            normal_df.to_pickle(pkl_normal_file_path)            
        gc.collect()

        # read pickle or raw dataset file with pandas
        if os.path.isfile(pkl_anom_file_path):
            print("## PKL Anomalous File: ", pkl_anom_file_path)
            anom_df = pd.read_pickle(pkl_anom_file_path)
        else:
            print("## Raw Anomalous File: ", raw_anom_file_path)            
            anom_df = pd.read_csv(raw_anom_file_path, low_memory=True, header=0, dtype=column_types)
            anom_df = data_cleasing(anom_df)
            anom_df.to_pickle(pkl_anom_file_path)
        gc.collect()

        if all_normal_df is None:
            all_normal_df = normal_df
        else:
            all_normal_df.append(normal_df)

        if all_anom_df is None:
            all_anom_df = anom_df
        else:
            all_anom_df.append(anom_df)
        gc.collect()

    # drop features
    all_normal_df.drop(drop_raw_features[features_key], axis=1, inplace=True)
    all_anom_df.drop(drop_raw_features[features_key], axis=1, inplace=True)

    # data merge and splitting
    train_df, cv_df, test_df, cv_label_df, test_label_df = data_merge_splitting(all_normal_df, all_anom_df)

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
    test_label = test_label_df.astype(int).values

    # print results
    f1, Recall, Precision = get_classification_report(test_label_df.astype(int).values, pred_test_label)
    print('###[MB-KMeans][', features_key, '] Test. F1:', f1, ', Recall:', Recall, ', Precision:', Precision)
print("--- %s seconds ---" % (time.time() - start_time))