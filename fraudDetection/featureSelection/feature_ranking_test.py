from __future__ import division

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE


# creates dictionary for data ranking
def rank_to_dict(_ranks, _names, order=1):
	minmax = MinMaxScaler()
	_ranks = minmax.fit_transform(order*np.array([_ranks]).T).T[0]
	_ranks = map(lambda x: round(x, 2), _ranks)
	return dict(zip(_names, _ranks))


# print results of feature selection avaliation
def print_feature_ranking(_data, _class, _names, _alg, _alg_name):

	ranks = {}

	# self defined algorithm
	_alg.fit(_data, _class)
	ranks[_alg_name] = rank_to_dict(np.abs(lr.coef_), _names)

	# Univariate linear regression tests.
	f, pval = f_regression(_data, _class, center=True)
	ranks["f_reg"] = rank_to_dict(f, _names)

	# Linear least squares with l2 regularization.
	ridge = Ridge(alpha=7)
	ridge.fit(_data, _class)
	ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), _names)

	# Linear Model trained with L1 prior as regularizer
	lasso = Lasso(alpha=.05)
	lasso.fit(_data, _class)
	ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), _names)

	# Randomized Lasso
	# This is known as stability selection. In short, features selected more often are considered good features.
	rlasso = RandomizedLasso(alpha=0.04)
	rlasso.fit(_data, _class)
	ranks["Stabi"] = rank_to_dict(np.abs(rlasso.scores_), _names)

	# Feature ranking with recursive feature elimination.
	rfe = RFE(_alg, n_features_to_select=2)
	rfe.fit(_data, _class)
	ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), _names, order=-1)

	# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the
	# dataset and use averaging to improve the predictive accuracy and control over-fitting
	rf = RandomForestRegressor()
	rf.fit(_data, _class)
	ranks["RandF"] = rank_to_dict(rf.feature_importances_, _names)

	# Maximal Information-based Nonparametric Exploration
	mine = MINE()
	mic_scores = []
	for i in range(_data.shape[1]):
		mine.compute_score(_data[:, i], _class)
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


np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + np.random.normal(0, 1))
# Add 3 additional correlated variables (correlated with X1-X3)
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))
names = ["x%s" % n for n in range(1, 15)]

lr = LinearRegression(normalize=True)

print_feature_ranking(X, Y, names, lr, "LinReg")