## evaluates classification performance of some algorithms for FCS classification
from sklearn import preprocessing
from numpy import genfromtxt
from sklearn import datasets, neighbors, linear_model, naive_bayes, lda,svm

# read data
target = genfromtxt('data/class_target', delimiter='\t')
data = genfromtxt('data/class_data_raw', delimiter='\t')

# data sampling
n_samples = len(data)
data_train = data[:int(.9 * n_samples)]
target_train = target[:int(.9 * n_samples)]
data_test = data[int(.9 * n_samples):]
target_test = target[int(.9 * n_samples):]

# classfication scores
print('# Classification scores:')
print('KNN: %f' % neighbors.KNeighborsClassifier().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.ElasticNet: %f' % linear_model.ElasticNet().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.ElasticNetCV: %f' % linear_model.ElasticNetCV().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.Lars: %f' % linear_model.Lars().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.Lasso: %f' % linear_model.Lasso().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.LassoCV: %f' % linear_model.LassoCV().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.LassoLars: %f' % linear_model.LassoLars().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.LassoLarsIC: %f' % linear_model.LassoLarsIC().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.LinearRegression: %f' % linear_model.LinearRegression().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.LogisticRegression: %f' % linear_model.LogisticRegression().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.OrthogonalMatchingPursuit: %f' % linear_model.OrthogonalMatchingPursuit().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.PassiveAggressiveClassifier: %f' % linear_model.PassiveAggressiveClassifier().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.PassiveAggressiveRegressor: %f' % linear_model.PassiveAggressiveRegressor().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.Perceptron: %f' % linear_model.Perceptron().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.Ridge: %f' % linear_model.Ridge().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.RidgeClassifier: %f' % linear_model.RidgeClassifier().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.RidgeClassifierCV: %f' % linear_model.RidgeClassifierCV().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.RidgeCV: %f' % linear_model.RidgeCV().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.SGDClassifier: %f' % linear_model.SGDClassifier().fit(data_train, target_train).score(data_test, target_test))
print('linear_model.SGDRegressor: %f' % linear_model.SGDRegressor().fit(data_train, target_train).score(data_test, target_test))
print('naive_bayes.MultinomialNB: %f' % naive_bayes.MultinomialNB().fit(data_train, target_train).score(data_test, target_test))
print('lda.LDA: %f' % lda.LDA().fit(data_train, target_train).score(data_test, target_test))
print('svm.SVR: %f' % svm.SVR().fit(data_train, target_train).score(data_test, target_test))
print('svm.SVC: %f' % svm.SVC(kernel='linear').fit(data_train, target_train).score(data_test, target_test))
print('svm.LinearSVC: %f' % svm.LinearSVC().fit(data_train, target_train).score(data_test, target_test))