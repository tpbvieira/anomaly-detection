import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import skew, kurtosis
from sklearn.metrics import pairwise_distances, f1_score

# from rpca_versions.rpca_kastnerkyle import robust_pca
# from rpca_versions.rpca_dganguli import robust_pca
# from rpca_versions.rpca_dfm import robust_pca
# from rpca_versions.rpca_bmcfee import robust_pca
# from rpca_versions.rpca_nwbirnie import robust_pca

from tensorly.decomposition.robust_decomposition import robust_pca
from hyperspy.learn.rpca import orpca


def fit(data, m_reg_J=1):
    """
    Robust PCA based estimation of mean, covariance, skewness and kurtosis.
    
    :param data: MxN matrix with M observations and N features, where M>N 
    :param m_reg_J: regularization. Default value is 1
    :return: 
        L: array-like, shape (m_obserations, n_features,)
            Robust data
        
        rob_mean:
        rob_cov:
        rob_dist:
        rob_precision:
        rob_skew:
        rob_skew_dist:
        rob_kurt:
        rob_kurt_dist:
    """

    L, S = robust_pca(data, reg_J=m_reg_J)

    rob_mean = L.mean(axis=0)
    rob_cov = pd.DataFrame(L).cov()
    rob_precision = linalg.pinvh(rob_cov)
    rob_dist = (np.dot(L, rob_precision) * L).sum(axis=1)

    rob_skew = skew(L, axis=0, bias=True)
    rob_skew_dist = (np.dot(L - rob_skew, rob_precision) * (L - rob_skew)).sum(axis=1)

    rob_kurt = kurtosis(L, axis=0, fisher=True, bias=True)
    rob_kurt_dist = (np.dot(L - rob_kurt, rob_precision) * (L - rob_kurt)).sum(axis=1)

    return L, rob_mean, rob_cov, rob_dist, rob_precision, rob_skew, rob_skew_dist, rob_kurt, rob_kurt_dist


def cv_location_contamination(cv_df, cv_labels, location, precision):
    """

    :param cv_df: cross-validation data frame
    :param cv_labels: labels to evaluate prediction performance by contamination
    :param location: mean vector
    :param precision: inverse of covariance matrix
    :return: For all tested contamination rates, returns the rate in which the best F1-score were achieved.
    """

    contamination = round(0.00, 2)
    contamination_prediction_list = []
    labels = np.array(cv_labels)

    for i in range(40):
        contamination += 0.01
        contamination = round(contamination, 2)
        pred_label = predict_by_location_contamination(cv_df, location, precision, contamination)
        contamination_prediction_list.append((contamination, f1_score(labels, pred_label)))

    contamination_prediction_list.sort(key=lambda tup: tup[1], reverse=True)
    contamination_best_f1 = contamination_prediction_list[0][0]

    return contamination_best_f1


def cv_location_threshold(cv_df, cv_labels, location, precision, dist):
    """

    :param cv_df: cross-validation data frame
    :param cv_labels: labels to evaluate prediction performance by contamination
    :param location:
    :param precision:
    :return: For all tested contamination rates, returns the rate in which the best F1-score were achieved.
    """

    threshold_prediction_list = []
    labels = np.array(cv_labels)
    min_dist = min(dist)
    max_dist = max(dist)

    for m_threshold in np.linspace(min_dist, max_dist, 40):
        pred_label = predict_by_location_threshold(cv_df, location, precision, m_threshold)
        threshold_prediction_list.append((m_threshold, f1_score(labels, pred_label)))

    threshold_prediction_list.sort(key=lambda tup: tup[1], reverse=True)
    best_threshold = threshold_prediction_list[0][0]

    return best_threshold


def cv_skewness_contamination(cv_df, cv_labels, skewness, precision):
    """

    :param cv_df: cross-validation data frame
    :param cv_labels: labels to evaluate prediction performance by contamination
    :param skewness:
    :param precision:
    :return: For all tested contamination rates, returns the rate in which the best F1-score were achieved.
    """

    contamination = round(0.00, 2)
    contamination_prediction_list = []
    actual_anomalies = np.array(cv_labels)
    for i in range(40):
        contamination += 0.01
        contamination = round(contamination, 2)
        pred_label = predict_by_skewness_contamination(cv_df, precision, skewness, contamination)
        contamination_prediction_list.append((contamination, f1_score(actual_anomalies, pred_label)))

    contamination_prediction_list.sort(key=lambda tup: tup[1], reverse=True)
    best_contamination = contamination_prediction_list[0][0]

    return best_contamination


def cv_skewness_threshold(cv_df, cv_labels, skewness, precision, skew_dist):
    """

    :param cv_df:
    :param cv_labels: labels to evaluate prediction performance by contamination
    :param skewness:
    :param precision:
    :param skew_dist:
    :return:
    """

    threshold_prediction_list = []
    actual_anomalies = np.array(cv_labels)
    min_dist = min(skew_dist)
    max_dist = max(skew_dist)

    for m_threshold in np.linspace(min_dist, max_dist, 40):
        pred_label = predict_by_skewness_threshold(cv_df, precision, skewness, m_threshold)
        threshold_prediction_list.append((m_threshold, f1_score(actual_anomalies, pred_label)))

    threshold_prediction_list.sort(key=lambda tup: tup[1], reverse=True)
    best_threshold = threshold_prediction_list[0][0]

    return best_threshold


def cv_kurtosis_contamination(cv_df, cv_labels, m_kurtosis, precision):
    """

    :param df: cross-validation data frame
    :param location:
    :param precision:
    :return: For all tested contamination rates, returns the rate in which the best F1-score were achieved.
    """

    contamination = round(0.00, 2)
    contamination_prediction_list = []
    actual_anomalies = np.array(cv_labels)
    for i in range(40):
        contamination += 0.01
        contamination = round(contamination, 2)
        pred_label = predict_by_kurtosis_contamination(cv_df, precision, m_kurtosis, contamination)
        contamination_prediction_list.append((contamination, f1_score(actual_anomalies, pred_label)))

    contamination_prediction_list.sort(key=lambda tup: tup[1], reverse=True)
    best_contamination = contamination_prediction_list[0][0]

    return best_contamination


def cv_kurtosis_threshold(cv_df, cv_labels, kurtosis, precision, kurt_dist):
    """

    :param cv_df:
    :param cv_labels: labels to evaluate prediction performance by contamination
    :param kurtosis:
    :param precision:
    :param kurt_dist:
    :return:
    """

    threshold_prediction_list = []
    actual_anomalies = np.array(cv_labels)
    min_dist = min(kurt_dist)
    max_dist = max(kurt_dist)

    for m_threshold in np.linspace(min_dist, max_dist, 40):
        pred_label = predict_by_kurtosis_threshold(cv_df, precision, kurtosis, m_threshold)
        threshold_prediction_list.append((m_threshold, f1_score(actual_anomalies, pred_label)))

    threshold_prediction_list.sort(key=lambda tup: tup[1], reverse=True)
    best_threshold = threshold_prediction_list[0][0]

    return best_threshold


def predict_by_location_contamination(test_df, location, precision, contamination):
    """

    :param test_df:
    :param location:
    :param precision:
    :param contamination:
    :return:
    """

    pred_label = np.full(test_df.shape[0], 0, dtype=int)
    if contamination is not None:
        # malhalanobis distance
        mahal_dist = pairwise_distances(test_df, location[np.newaxis, :], metric='mahalanobis', VI=precision)
        mahal_dist = np.reshape(mahal_dist, (len(test_df),)) ** 2  #MD squared
        # detect outliers
        contamination_threshold = np.percentile(mahal_dist,  100. * (1. - contamination))
        pred_label[mahal_dist > contamination_threshold] = 1
    else:
        raise NotImplementedError("You must provide a contamination rate.")

    return pred_label


def predict_by_location_centered_contamination(X, location, precision, contamination):
    """

    :param X:
    :param location:
    :param precision:
    :param contamination:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)
    if contamination is not None:
        # malhalanobis distance
        X = X - location
        mahal_dist = pairwise_distances(X, location[np.newaxis, :], metric='mahalanobis', VI=precision)
        mahal_dist = np.reshape(mahal_dist, (len(X),)) ** 2  #MD squared
        # detect outliers
        contamination_threshold = np.percentile(mahal_dist,  100. * (1. - contamination))
        pred_label[mahal_dist > contamination_threshold] = 1
    else:
        raise NotImplementedError("You must provide a contamination rate.")

    return pred_label


def predict_by_location_threshold(X, location, precision, threshold):
    """

    :param X:
    :param location:
    :param precision:
    :param threshold:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X, location[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X),)) ** 2  #MD squared
    # detect outliers
    pred_label[mahal_dist > threshold] = 1

    return pred_label


def predict_by_skewness_contamination(X, precision, skewness, contamination):
    """

    :param X:
    :param precision:
    :param skewness:
    :param contamination:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X, skewness[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X),)) ** 2  # MD squared
    pred_skew_dist = -mahal_dist

    # detect outliers
    contamination_threshold = np.percentile(pred_skew_dist, 100. * contamination)
    pred_label[pred_skew_dist <= contamination_threshold] = 1

    return pred_label


def predict_by_skewness_centered_contamination(X, precision, skewness, contamination):
    """

    :param X:
    :param precision:
    :param skewness:
    :param contamination:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # skewness of the data
    X_skew = X - skew(X, axis=0, bias=True)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X_skew, skewness[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X_skew),)) ** 2 #MD squared
    pred_skew_dist = -mahal_dist

    # detect outliers
    contamination_threshold = np.percentile(pred_skew_dist, 100. * contamination)
    pred_label[pred_skew_dist <= contamination_threshold] = 1

    return pred_label


def predict_by_skewness_threshold(X, precision, skewness, threshold):
    """

    :param X:
    :param precision:
    :param skewness:
    :param threshold:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X, skewness[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X),)) ** 2  # MD squared
    pred_skew_dist = -mahal_dist

    # detect outliers
    pred_label[pred_skew_dist <= threshold] = 1

    return pred_label


def predict_by_skewness_centered_threshold(X, precision, skewness, threshold):
    """

    :param X:
    :param precision:
    :param skewness:
    :param threshold:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # skewness of the data
    X_skew = X - skew(X, axis=0, bias=True)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X_skew, skewness[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X_skew),)) ** 2  # MD squared
    pred_skew_dist = -mahal_dist

    # detect outliers
    pred_label[pred_skew_dist <= threshold] = 1

    return pred_label


def predict_by_kurtosis_contamination(X, precision, m_kurtosis, contamination):
    """

    :param X:
    :param precision:
    :param m_kurtosis	:
    :param contamination:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X, m_kurtosis[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X),)) ** 2  # MD squared
    pred_kurt_dist = -mahal_dist

    # detect outliers
    contamination_threshold = np.percentile(pred_kurt_dist, 100. * contamination)
    pred_label[pred_kurt_dist <= contamination_threshold] = 1

    return pred_label


def predict_by_kurtosis_centered_contamination(X, precision, m_kurtosis, contamination):
    """

    :param X:
    :param precision:
    :param m_kurtosis	:
    :param contamination:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # m_kurtosis	 of the data
    X_kurt = X - kurtosis(X, axis=0, bias=True)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X_kurt, m_kurtosis[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X_kurt),)) ** 2 #MD squared
    pred_kurt_dist = -mahal_dist

    # detect outliers
    contamination_threshold = np.percentile(pred_kurt_dist, 100. * contamination)
    pred_label[pred_kurt_dist <= contamination_threshold] = 1

    return pred_label


def predict_by_kurtosis_threshold(X, precision, m_kurtosis, threshold):
    """

    :param X:
    :param precision:
    :param m_kurtosis	:
    :param threshold:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X, m_kurtosis[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X),)) ** 2  # MD squared
    pred_kurt_dist = -mahal_dist

    # detect outliers
    pred_label[pred_kurt_dist <= threshold] = 1

    return pred_label


def predict_by_kurtosis_centered_threshold(X, precision, m_kurtosis, threshold):
    """

    :param X:
    :param precision:
    :param m_kurtosis	:
    :param threshold:
    :return:
    """

    pred_label = np.full(X.shape[0], 0, dtype=int)

    # m_kurtosis	 of the data
    X_kurt = X - kurtosis(X, axis=0, bias=True)

    # malhalanobis distance
    mahal_dist = pairwise_distances(X_kurt, m_kurtosis[np.newaxis, :], metric='mahalanobis', VI=precision)
    mahal_dist = np.reshape(mahal_dist, (len(X_kurt),)) ** 2  # MD squared
    pred_kurt_dist = -mahal_dist

    # detect outliers
    pred_label[pred_kurt_dist <= threshold] = 1

    return pred_label
