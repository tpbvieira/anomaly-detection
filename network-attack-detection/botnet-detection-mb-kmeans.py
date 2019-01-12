# coding=utf-8
import os, gc, time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.cluster import MiniBatchKMeans


def get_classification_report(y_test, y_predic):
    f1 = f1_score(y_test, y_predic, average = "binary")
    Recall = recall_score(y_test, y_predic, average = "binary")
    Precision = precision_score(y_test, y_predic, average = "binary")
    return f1, Recall,Precision


def data_splitting(df, drop_feature):
    # drop non discriminant features
    df.drop(drop_feature, axis=1, inplace=True)

    # split into normal and anomaly
    df_l1 = df[df["Label"] == 1]
    df_l0 = df[df["Label"] == 0]
    gc.collect()

    # Length and indexes
    anom_len = len(df_l1)  # total number of anomalous flows
    anom_train_end = anom_len // 2  # 50% of anomalous for training
    anom_cv_start = anom_train_end + 1  # 50% of anomalous for testing
    norm_len = len(df_l0)  # total number of normal flows
    norm_train_end = (norm_len * 60) // 100  # 60% of normal for training
    norm_cv_start = norm_train_end + 1  # 20% of normal for cross validation
    norm_cv_end = (norm_len * 80) // 100  # 20% of normal for cross validation
    norm_test_start = norm_cv_end + 1  # 20% of normal for testing

    # anomalies split data
    anom_cv_df = df_l1[:anom_train_end]  # 50% of anomalies59452
    anom_test_df = df_l1[anom_cv_start:anom_len]  # 50% of anomalies
    gc.collect()

    # normal split data
    norm_train_df = df_l0[:norm_train_end]  # 60% of normal
    norm_cv_df = df_l0[norm_cv_start:norm_cv_end]  # 20% of normal
    norm_test_df = df_l0[norm_test_start:norm_len]  # 20% of normal
    gc.collect()

    # CV and test data. train data is norm_train_df
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
    norm_train_df = norm_train_df.drop(labels=["Label"], axis=1)
    cv_df = cv_df.drop(labels=["Label"], axis=1)
    test_df = test_df.drop(labels=["Label"], axis=1)

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

column_types = {
    'StartTime': 'str',
    'Dur': 'float32',
    'Proto': 'str',
    'SrcAddr': 'str',
    'Sport': 'str',
    'Dir': 'str',
    'DstAddr': 'str',
    'Dport': 'str',
    'State': 'str',
    'sTos': 'float16',
    'dTos': 'float16',
    'TotPkts': 'uint32',
    'TotBytes': 'uint32',
    'SrcBytes': 'uint32',
    'Label': 'str'}

# feature selection
drop_features = {
    'drop_features00': []
    # 'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes', 'Proto'],
    # 'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes'],
    # 'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'Proto'],
    # 'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
}

raw_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/network-attack-detection/BinetflowTrainer-master/pkl/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)

pkl_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/network-attack-detection/BinetflowTrainer-master/pkl/')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_features.items():

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
        norm_train_df, cv_df, test_df, cv_label_df, test_label_df = data_splitting(df, drop_features[features_key])

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