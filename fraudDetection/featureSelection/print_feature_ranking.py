from __future__ import division

import numpy as np

from sklearn.feature_selection import RFE, RFECV, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, RandomizedLasso
from minepy import MINE


# creates dictionary for data ranking
def rank_to_dict(_ranks, _names, order=1):
	minmax = MinMaxScaler()
	_ranks = minmax.fit_transform(order * np.array([_ranks]).T).T[0]
	_ranks = map(lambda x: round(x, 2), _ranks)
	return dict(zip(_names, _ranks))


# print results of feature selection avaliation
def print_feature_ranking(_data, _target, _names, _model, _model_name):
	ranks = {}

	# self defined algorithm
	_model.fit(_data, _target)
	ranks[_model_name] = rank_to_dict(np.abs(_model.coef_.ravel()), _names)

	# Univariate linear regression
	f, pval = f_regression(_data, _target, center=True)
	ranks["f_reg"] = rank_to_dict(f, _names)

	# Linear least squares with l2 regularization
	ridge = Ridge(alpha=7)
	ridge.fit(_data, _target)
	ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), _names)

	# Linear Model trained with L1 prior as regularizer
	lasso = Lasso(alpha=.05)
	lasso.fit(_data, _target)
	ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), _names)

	# Randomized Lasso
	# This is known as stability selection. In short, features selected more often are considered good features.
	rlasso = RandomizedLasso(alpha=0.001)
	rlasso.fit(_data, _target)
	ranks["Stabi"] = rank_to_dict(np.abs(rlasso.scores_), _names)

	# Feature ranking with recursive feature elimination
	rfe = RFE(_model, n_features_to_select=2)
	rfe.fit(_data, _target)
	ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), _names, order=-1)
	print("[RFE] Optimal number of features : %d" % rfe.n_features_)

	# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the
	# dataset and use averaging to improve the predictive accuracy and control over-fitting
	rf = RandomForestRegressor()
	rf.fit(_data, _target)
	ranks["RandF"] = rank_to_dict(rf.feature_importances_, _names)

	# Maximal Information-based Nonparametric Exploration
	mine = MINE()
	mic_scores = []
	for i in range(_data.shape[1]):
		mine.compute_score(_data[:, i], _target)
		m = mine.mic()
		mic_scores.append(m)
	ranks["MIC"] = rank_to_dict(mic_scores, _names)

	# mean score for each feature
	means = {}
	for name in _names:
		means[name] = np.mean([ranks[method][name] for method in ranks.keys()], dtype=np.float)
	methods = sorted(ranks.keys())
	ranks["Mean"] = means
	methods.append("Mean")

	print("\t%s" % "\t".join(methods))
	for name in _names:
		print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))
