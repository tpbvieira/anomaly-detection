# Create a signal with only 2 useful dimensions
from sklearn import decomposition
from sklearn import preprocessing
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

org = genfromtxt('org', delimiter='\t')
fcs = genfromtxt('fcs', delimiter='\t')
data = genfromtxt('data', delimiter='\t')
convdata = genfromtxt('convdata', delimiter='\t')
normalizer = preprocessing.Normalizer().fit(data)
normdata = normalizer.transform(data)

# raw 
index = np.power(org[:,1] * 20, 2)
pca = decomposition.PCA()
pca.fit(data)
np.savetxt('eigenvalues.txt', pca.explained_variance_)
np.savetxt('eigenvalues_ratio.txt', pca.explained_variance_ratio_)
pcs = pca.explained_variance_ratio_
pca.n_components = 2
reduced = pca.fit_transform(data)
# PCs_FCS
plt.scatter(pcs * 100, fcs[:,1], s=pcs * 10000, c=fcs[:,1], alpha=0.1)
plt.title('Classification of PCs into FCS or not (Raw Data)')
plt.xlabel('PCs (%)')
plt.ylabel('FCS (1 = True, -1 = False)')
plt.grid()
plt.savefig('raw_pcs_fcs.png', format='png')
plt.clf()
# Best Institutions_PC1_PC2
plt.scatter(reduced[:,0], reduced[:,1], s=index, c=index, alpha=0.5)
plt.title('Two Principal Components (Raw Data) ordered by OrgClassification')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.savefig('raw_pcs.png', format='png')
plt.clf()


# converted
index = np.power(org[:,1] * 20, 2)
pca = decomposition.PCA()
pca.fit(convdata)
pcs = pca.explained_variance_ratio_
pca.n_components = 2
reduced = pca.fit_transform(convdata)
plt.scatter(reduced[:,0], reduced[:,1], s=index, c=index, alpha=0.5)
plt.title('Two Principal Components (Converted Data)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.savefig('conv_pcs.png', format='png')
plt.clf()
plt.scatter(pcs * 100, fcs[:,1], s=pcs * 10000, c=fcs[:,1], alpha=0.1)
plt.title('Classification of PCs into FCS or not (Converted Data)')
plt.xlabel('PCs (%)')
plt.ylabel('FCS (1 = True, -1 = False)')
plt.grid()
plt.savefig('conv_pcs_fcs.png', format='png')
plt.clf()

# normalized
index = np.power(org[:,1] * 20, 2)
pca = decomposition.PCA()
pca.fit(normdata)
pcs = pca.explained_variance_ratio_
pca.n_components = 2
reduced = pca.fit_transform(normdata)
plt.scatter(reduced[:,0], reduced[:,1], s=index, c=index, alpha=0.5)
plt.title('Two Principal Components (Normalized Data)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.savefig('norm_pcs.png', format='png')
plt.clf()
plt.scatter(pcs * 100, fcs[:,1], s=pcs * 10000, c=fcs[:,1], alpha=0.1)
plt.title('Classification of PCs into FCS or not (Normalized Data)')
plt.xlabel('PCs (%)')
plt.ylabel('FCS (1 = True, -1 = False)')
plt.grid()
plt.savefig('norm_pcs_fcs.png', format='png')
plt.clf()
