# coding:utf-8
## Principal Component Analysis for iGovTI-2012

from sklearn import decomposition, preprocessing
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from mpl_toolkits.axes_grid1 import make_axes_locatable

## settings
org_igovti = genfromtxt('data/organizations_igovti.txt', delimiter='\t')
fcs = genfromtxt('data/fcs', delimiter='\t')
data = genfromtxt('data/data', delimiter='\t')
cmap = cm.get_cmap('jet', 30)
# convdata = genfromtxt('convdata', delimiter='\t')

## Plots a Covariance Heatmap
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
cax = ax1.imshow(pd.DataFrame(data).cov(), interpolation="nearest", cmap=cm.Blues)
ax1.grid(True)
plt.title("Covariance matrix of iGovTI questions")
# labels = dataframe.columns.tolist()
# ax1.set_xticklabels(labels, fontsize=13, rotation=45)
# ax1.set_yticklabels(labels, fontsize=13)
divider = make_axes_locatable(plt.gca())
fig.colorbar(cax, cax=divider.append_axes("right", size="5%", pad=0.05))
plt.savefig('results/raw_igovti_covariance.eps', format='eps', dpi=600)
plt.show()

## Plots a Correlation Heatmap
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
cax = ax1.imshow(pd.DataFrame(data).corr(), interpolation="nearest", cmap=cm.Blues)
ax1.grid(True)
plt.title("Correlation matrix of iGovTI questions")
# labels = dataframe.columns.tolist()
# ax1.set_xticklabels(labels, fontsize=13, rotation=45)
# ax1.set_yticklabels(labels, fontsize=13)
divider = make_axes_locatable(plt.gca())
fig.colorbar(cax, cax=divider.append_axes("right", size="5%", pad=0.05))
plt.savefig('results/raw_igovti_correlation.eps', format='eps', dpi=600)
plt.show()

## raw_data PCA
pca = decomposition.PCA()
pca.fit(data)
eigenvalues = pca.explained_variance_
eigenvalues_ratio = pca.explained_variance_ratio_
np.savetxt('results/raw_eigenvalues.txt', eigenvalues)
np.savetxt('results/raw_eigenvalues_ratio.txt', eigenvalues_ratio)
pca.n_components = 2
reduced = pca.fit_transform(data)

## Plot raw_data Variance ECDf 
ecdf = ECDF(eigenvalues)
plt.title(u"Empirical CDF")
plt.ylabel(u"Cumulative Variance")
plt.xlabel(u"Variance")
plt.plot(ecdf.x, ecdf.y)
plt.grid()
plt.savefig('results/raw_variance_ecdf.eps', format='eps', dpi=600)
plt.show()
plt.clf()

## Plot iGovTI Ranking for PC1_PC2 of raw_data
igovti_index = np.power(org_igovti[:,1] * 20, 2)
sc = plt.scatter(reduced[:,0], reduced[:,1], s=igovti_index, c=igovti_index, alpha=0.5, cmap=cmap)
plt.title(u"iGovTI Ranking")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.colorbar(sc)
plt.savefig('results/raw_igovti_ranking_pc2.eps', format='eps', dpi=600)
plt.show()
plt.clf()

# ## norm_data PCA
# normdata = preprocessing.Normalizer().fit_transform(data)
# pca = decomposition.PCA()
# pca.fit(normdata)
# eigenvalues = pca.explained_variance_
# eigenvalues_ratio = pca.explained_variance_ratio_
# np.savetxt('norm_eigenvalues.txt', eigenvalues)
# np.savetxt('norm_eigenvalues_ratio.txt', eigenvalues_ratio)
# pca.n_components = 2
# reduced = pca.fit_transform(normdata)

# ## Plot norm_data Variance ECDf 
# ecdf = ECDF(eigenvalues)
# plt.title(u"Empirical CDF")
# plt.ylabel(u"Cumulative Variance")
# plt.xlabel(u"Variance")
# plt.plot(ecdf.x, ecdf.y)
# plt.grid()
# plt.savefig('results/norm_variance_ecdf.eps', format='eps', dpi=600)
# plt.show()
# plt.clf()

# ## Plot iGovTI Ranking for PC1_PC2 of norm_data
# plt.scatter(reduced[:,0], reduced[:,1], s=igovti_index, c=igovti_index, alpha=0.5, cmap=cmap)
# plt.title(u"iGovTI Ranking")
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.grid()
# plt.savefig('results/norm_igovti_ranking_pc2.eps', format='eps', dpi=600)
# plt.show()
# plt.clf()

## Recursive Feature Elimination
# from sklearn import datasets
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# # create a base classifier used to evaluate a subset of attributes
# model = LogisticRegression()

# # create the RFE model and select 3 attributes
# rfe = RFE(model, 3)
# rfe = rfe.fit(dataset.data, dataset.target)
# # summarize the selection of the attributes
# print(rfe.support_)
# print(rfe.ranking_)