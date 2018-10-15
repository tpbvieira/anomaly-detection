import pandas as pd
import numpy as np
import os
import datetime as dt
import seaborn as sns
import matplotlib.gridspec as gridspec
import ipaddress
import random as rnd
import plotly.graph_objs as go
import lime
import lime.lime_tabular
import itertools
from pandas.tools.plotting import scatter_matrix
from functools import reduce
from numpy import genfromtxt
from scipy import linalg
from scipy.stats import multivariate_normal
from sklearn import preprocessing, mixture
from sklearn.metrics import classification_report, average_precision_score, f1_score, recall_score, precision_score
from sklearn.cluster import KMeans
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def data_cleasing(df):
	# data cleasing, feature engineering and save clean data into pickles

	print('### Data Cleasing and Feature Engineering')
	le = preprocessing.LabelEncoder()

	# [Protocol] - Discard ipv6-icmp and categorize
	df['Proto'] = df['Proto'].fillna('-')
	df['Proto'] = le.fit_transform(df['Proto'])
	le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

	# [Label] - Categorize
	anomalies = df.Label.str.contains('Botnet')
	normal = np.invert(anomalies)
	df.loc[anomalies, 'Label'] = int(1)
	df.loc[normal, 'Label'] = int(0)
	df['Label'] = pd.to_numeric(df['Label'])

	# [Dport] - replace NaN with 0 port number
	df['Dport'] = df['Dport'].fillna('0')
	df['Dport'] = df['Dport'].apply(lambda x: int(x,0))

	# [sport] - replace NaN with 0 port number
	df['Sport'] = df['Sport'].fillna('0')
	df['Sport'] = df['Sport'].apply(lambda x: int(x,0))

	# [sTos] - replace NaN with "10" and convert to int
	df['sTos'] = df['sTos'].fillna('10')
	df['sTos'] = df['sTos'].astype(int)

	# [dTos] - replace NaN with "10" and convert to int
	df['dTos'] = df['dTos'].fillna('10')
	df['dTos'] = df['dTos'].astype(int)

	# [State] - replace NaN with "-" and categorize
	df['State'] = df['State'].fillna('-')
	df['State'] = le.fit_transform(df['State'])
	le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

	# [Dir] - replace NaN with "-" and categorize
	df['Dir'] = df['Dir'].fillna('-')
	df['Dir'] = le.fit_transform(df['Dir'])
	le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

	# [SrcAddr] Extract subnet features and categorize
	df['SrcAddr'] = df['SrcAddr'].fillna('0.0.0.0')

	# [DstAddr] Extract subnet features
	df['DstAddr'] = df['DstAddr'].fillna('0.0.0.0')

	# [StartTime] - Parse to datatime, reindex based on StartTime, but first drop the ns off the time stamps
	df['StartTime'] = df['StartTime'].apply(lambda x: x[:19])
	df['StartTime'] = pd.to_datetime(df['StartTime'])
	df = df.set_index('StartTime')

	# save clean data into pickles
	df.to_pickle(pkl_file_path)  # where to save it, usually as a .pkl

	return df


def classify_ip(ip):
	'''
	str ip - ip address string to attempt to classify. treat ipv6 addresses as N/A
	'''
	try:
		ip_addr = ipaddress.ip_address(ip)
		if isinstance(ip_addr, ipaddress.IPv6Address):
			return 'ipv6'
		elif isinstance(ip_addr, ipaddress.IPv4Address):
			# split on .
			octs = ip_addr.exploded.split('.')
			if 0 < int(octs[0]) < 127: return 'A'
			elif 127 < int(octs[0]) < 192: return 'B'
			elif 191 < int(octs[0]) < 224: return 'C'
			else: return 'N/A'
	except ValueError:
		return 'N/A'


def avg_duration(x):
	return np.average(x)


def n_dports_gt1024(x):
	if x.size == 0: return 0
	return reduce((lambda a,b: a+b if b>1024 else a),x)
n_dports_gt1024.__name__ = 'n_dports>1024'


def n_dports_lt1024(x):
	if x.size == 0: return 0
	return reduce((lambda a,b: a+b if b<1024 else a),x)
n_dports_lt1024.__name__ = 'n_dports<1024'


def n_sports_gt1024(x):
	if x.size == 0: return 0
	return reduce((lambda a,b: a+b if b>1024 else a),x)
n_sports_gt1024.__name__ = 'n_sports>1024'


def n_sports_lt1024(x):
	if x.size == 0: return 0
	return reduce((lambda a,b: a+b if b<1024 else a),x)
n_sports_lt1024.__name__ = 'n_sports<1024'


def label_atk_v_norm(x):
	for l in x:
		if l == 1: return 1
	return 0
label_atk_v_norm.__name__ = 'label'


def background_flow_count(x):
	count = 0
	for l in x:
		if l == 0: count += 1
	return count


def normal_flow_count(x):
	if x.size == 0: return 0
	count = 0
	for l in x:
		if l == 0: count += 1
	return count


def n_conn(x):
	return x.size


def n_tcp(x):
	count = 0
	for p in x:
		if p == 10: count += 1 # tcp == 10
	return count


def n_udp(x):
	count = 0
	for p in x:
		if p == 11: count += 1 # udp == 11
	return count


def n_icmp(x):
	count = 0
	for p in x:
		if p == 1: count += 1 # icmp == 1
	return count


def n_s_a_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'A': count += 1
	return count


def n_d_a_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'A': count += 1
	return count


def n_s_b_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'B': count += 1
	return count


def n_d_b_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'A': count += 1
	return count


def n_s_c_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'C': count += 1
	return count


def n_d_c_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'C': count += 1
	return count


def n_s_na_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'N/A': count += 1
	return count


def n_d_na_p_address(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'N/A': count += 1
	return count


def n_ipv6(x):
	count = 0
	for i in x:
		if classify_ip(i) == 'ipv6': count += 1
	return count


def print_classification_report(y_test, y_predic):
	f1 = f1_score(y_test, y_predic, average = "binary")
	Recall = recall_score(y_test, y_predic, average = "binary")
	Precision = precision_score(y_test, y_predic, average = "binary")
	print('\tF1 Score: ',f1,', Recall: ',Recall,', Precision: ,',Precision)


def data_splitting(m_df, m_drop_features):
	# Data splitting

	# drop non discriminant features
	m_df.drop(m_drop_features, axis =1, inplace = True)

	# split into normal and anomaly
	df_l1 = m_df[m_df["Label"] == 1]
	df_l0 = m_df[m_df["Label"] == 0]

	# Length and indexes
	norm_len = len(df_l0)
	anom_len = len(df_l1)
	anom_train_end = anom_len // 2
	anom_cv_start = anom_train_end + 1
	norm_train_end = (norm_len * 60) // 100
	norm_cv_start = norm_train_end + 1
	norm_cv_end = (norm_len * 80) // 100
	norm_test_start = norm_cv_end + 1

	# anomalies split data
	anom_cv_df = df_l1[:anom_train_end] # 50% of anomalies59452
	anom_test_df = df_l1[anom_cv_start:anom_len] # 50% of anomalies

	# normal split data
	norm_train_df = df_l0[:norm_train_end] # 60% of normal
	norm_cv_df = df_l0[norm_cv_start:norm_cv_end] # 2059452 % of normal
	norm_test_df = df_l0 [norm_test_start:norm_len] # 20% of normal

	# CV and test data. train data is norm_train_df
	cv_df = pd.concat([norm_cv_df, anom_cv_df], axis=0)
	test_df = pd.concat([norm_test_df, anom_test_df], axis=0)

	# labels
	cv_label = cv_df["Label"]
	test_label = test_df["Label"]

	# drop label
	norm_train_df = norm_train_df.drop(labels = ["Label"], axis = 1)
	cv_df = cv_df.drop(labels = ["Label"], axis = 1)
	test_df = test_df.drop(labels = ["Label"], axis = 1)

	return norm_train_df, cv_df, test_df, cv_label, test_label


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


# features
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
	'drop_features01':['SrcAddr','DstAddr','sTos','Sport','Proto','TotBytes','SrcBytes'],
	'drop_features02':['SrcAddr','DstAddr','sTos','Sport','TotBytes','SrcBytes'],
	'drop_features03':['SrcAddr','DstAddr','sTos','Sport','Proto','SrcBytes'],
	'drop_features04':['SrcAddr','DstAddr','sTos','Proto']
}

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/raw_fast/')
raw_directory = os.fsencode(raw_path)
raw_files = os.listdir(raw_directory)
print("## Directory: ", raw_directory)
print("## Files: ", raw_files)

# pickle files have the same names
pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/ctu-13/pkl_fast/')
pkl_directory = os.fsencode(pkl_path)

# for each feature set
for features_key, value in drop_features.items():

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
			raw_df = pd.read_csv(raw_file_path, low_memory=False, dtype={'Label':'str'})
			df = data_cleasing(raw_df)

		# data splitting
		train_df, cv_df, test_df, cv_label, test_label = data_splitting(df, drop_features[features_key])

		# Cross-Validation
    best_num_clusters, best_epsilon, best_f1, best_precision, best_recall = getBestByCV(train_df, cv_df, cv_label)
		print('###[KMeans][',features_key,'] Cross-Validation (best_num_clusters, best_epsilon, best_f1, best_precision, best_recall): ', best_num_clusters, best_epsilon, best_f1, best_precision, best_recall)

		# Training - estimate clusters (anomalous or normal) for training
		kmeans = KMeans(n_clusters = best_num_clusters).fit(train_df)

		# Test prediction
		test_clusters = kmeans.predict(test_df)
		test_clusters_centers = kmeans.cluster_centers_

		dist = [np.linalg.norm(x-y) for x,y in zip(test_df.as_matrix(), test_clusters_centers[test_clusters])]

		pred_test_label = np.array(dist)
		pred_test_label[dist >= np.percentile(dist, best_epsilon)] = 1
		pred_test_label[dist < np.percentile(dist, best_epsilon)] = 0

		print('###[KMeans][',features_key,'] Test')
		print_classification_report(test_label, pred_test_label)

		kmeans_test_label.extend(test_label.astype(int))			# append into global array
		kmeans_pred_test_label.extend(pred_test_label.astype(int))	# append into global array

	# print results
	print('###[KMeans][',features_key,'] Test Full dataset')
	print_classification_report(kmeans_test_label, kmeans_pred_test_label)
