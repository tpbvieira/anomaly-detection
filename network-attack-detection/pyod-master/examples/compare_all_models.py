# -*- coding: utf-8 -*-
"""Compare all detection algorithms by plotting decision boundaries and
the number of decision boundaries.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings
import time
warnings.filterwarnings("ignore")
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP
from botnet_detection_utils import data_splitting, unsupervised_data_splitting, drop_agg_features, get_classification_report, data_cleasing, column_types

# TODO: add neural networks, LOCI, SOS


raw_path = os.path.join('../../data/cicids2017/raw/pkl/')
raw_directory = os.fsencode(raw_path)
pkl_path = os.path.join('../../data/cicids2017/raw/pkl/')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

start_time = time.time()
# for each feature set
for features_key, value in drop_agg_features.items():
    for sample_file in file_list:

        # read pickle file with pandas or...
        pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
        if os.path.isfile(pkl_file_path):
            print("## Sample File: ", pkl_file_path)
            df = pd.read_pickle(pkl_file_path)

        # data splitting
        norm_train_df, cv_df, test_df, cv_label_df, test_label_df = data_splitting(df, drop_agg_features[features_key])
        # test_df = test_df[['background_flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address',
        #                   'n_sports<1024', 'n_s_na_p_address', 'n_s_c_p_address', 'normal_flow_count', 'n_dports<1024']]

        test_label_vc = test_label_df.value_counts()
        ones = test_label_vc.get(1)
        zeros = test_label_vc.get(0)
        outliers_fraction = ones/(ones + zeros)

        n_samples = test_label_df.size
        clusters_separation = [0]

        # Compare given detectors under given settings
        # Initialize the data
        # xx, yy = np.meshgrid(np.linspace(-7, 7, n_samples), np.linspace(-7, 7, n_samples))
        n_inliers = int((1. - outliers_fraction) * n_samples)
        n_outliers = int(outliers_fraction * n_samples)
        ground_truth = test_label_df.astype(np.int8)

        # initialize a set of detectors for LSCP
        detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15), LOF(n_neighbors=20), LOF(n_neighbors=25),
                         LOF(n_neighbors=30), LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45), LOF(n_neighbors=50)]

        # Show the statics of the data
        print('Number of inliers: %i' % n_inliers)
        print('Number of outliers: %i' % n_outliers)
        print('Ground truth shape is {shape}. Outlier are 1 and inliers are 0.\n'.format( shape=ground_truth.shape))
        print(ground_truth, '\n')

        random_state = np.random.RandomState(42)

        # Define nine outlier detection tools to be compared
        classifiers = {
            # 'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
            # 'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction,
            #                                                     check_estimator=False, random_state=random_state),
            # 'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,random_state=random_state),
            # 'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
            # 'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
            'Average KNN': KNN(method='mean',contamination=outliers_fraction),
            # 'Median KNN': KNN(method='median',contamination=outliers_fraction),
            'Local Outlier Factor (LOF)':LOF(n_neighbors=35, contamination=outliers_fraction),
            # 'Local Correlation Integral (LOCI)':LOCI(contamination=outliers_fraction),
            'Minimum Covariance Determinant (MCD)': MCD(contamination=outliers_fraction, random_state=random_state),
            'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
            'Principal Component Analysis (PCA)': PCA(contamination=outliers_fraction, random_state=random_state),
            # 'Stochastic Outlier Selection (SOS)': SOS(contamination=outliers_fraction),
            'Locally Selective Combination (LSCP)': LSCP(detector_list, contamination=outliers_fraction,
                                                         random_state=random_state)
        }

        # Show all detectors
        for i, clf in enumerate(classifiers.keys()):
            print('Model', i + 1, clf)

        # Fit the models with the generated data and compare model performances
        for i, offset in enumerate(clusters_separation):

            X = test_df.values

            # Fit the model
            plt.figure(figsize=(15, 12))
            for i, (clf_name, clf) in enumerate(classifiers.items()):
                print()
                print(i + 1, 'fitting', clf_name)
                # fit the data and tag outliers
                try:
                    clf.fit(X)
                    scores_pred = clf.decision_function(X) * -1
                    y_pred = clf.predict(X)
                    threshold = percentile(scores_pred, 100 * outliers_fraction)
                    n_errors = (y_pred != ground_truth).sum()

                    t_f1, t_Recall, t_Precision = get_classification_report(ground_truth, y_pred)
                    print('### F1:', t_f1, ', Recall:', t_Recall, ', Precision:', t_Precision)
                except:
                    print('Error!')

        #         # plot the levels lines and the points
        #         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        #         Z = Z.reshape(xx.shape)
        #         subplot = plt.subplot(3, 4, i + 1)
        #         subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        #         a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        #         subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        #         b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',s=20, edgecolor='k')
        #         c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',s=20, edgecolor='k')
        #         subplot.axis('tight')
        #         subplot.legend([a.collections[0], b, c],['learned decision function', 'true inliers', 'true outliers'],
        #             prop=matplotlib.font_manager.FontProperties(size=10),loc='lower right')
        #         subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        #         subplot.set_xlim((-7, 7))
        #         subplot.set_ylim((-7, 7))
        #     plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
        #     plt.suptitle("Outlier detection")
        # plt.savefig('ALL.png', dpi=300)
        # plt.show()
