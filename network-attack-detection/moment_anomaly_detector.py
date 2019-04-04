"""
Outlier detection based on robust moments.

This class provides a framework for outlier detection based on robust momentos. It consists in methods that can be
added to a covariance estimator in order to assess the outlying-ness of the observations of a data set. The proposed
"outlier detector" is based on robust skewness, kurtosis, location and covariance estimates.

"""
# Author: Thiago Vieira <tpbvieira@gmail.com>
#
# License: BSD 3 clause
#
# This is a fork of EllipticEnvelope class of Virgile Fritsch, which implements an anomaly detector based on classical
# Fast Minimum Covariance Determinant (Fast MCD). Fast NCD is an algorithm by Rousseeuw & Van Driessen described in
# (A Fast Algorithm for the Minimum Covariance Determinant Estimator, 1999, American Statistical Association and the
# American Society for Quality, TECHNOMETRICS)

import numpy as np
import scipy as sp
from scipy import linalg
from scipy.stats import skew,kurtosis
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import accuracy_score, pairwise_distances
from robust_moments import MomentMinCovDet


class MomentAnomalyDetector(MomentMinCovDet):
    """An object for detecting outliers in a skewed distributed and imbalanced dataset.

    Parameters
    ----------
    store_precision : boolean, optional (default=True)
        Specify if the estimated precision is stored.

    assume_centered : boolean, optional (default=False)
        If True, the support of robust location and covariance estimates
        is computed, and a covariance estimate is recomputed from it,
        without centering the data.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, the robust location and covariance are directly computed
        with the FastMCD algorithm without additional treatment.

    support_fraction : float in (0., 1.), optional (default=None)
        The proportion of points to be included in the support of the raw
        MCD estimate. If None, the minimum value of support_fraction will
        be used within the algorithm: `[n_sample + n_features + 1] / 2`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    raw_location_ : array-like, shape (n_features,)
        The raw robust estimated location before correction and re-weighting.

    raw_covariance_ : array-like, shape (n_features, n_features)
        The raw robust estimated covariance before correction and re-weighting.

    raw_support_ : array-like, shape (n_samples,)
        A mask of the observations that have been used to compute
        the raw robust estimates of location and shape, before correction
        and re-weighting.

    raw_skew1_ : array-like, shape (n_features,)
        The raw robust estimated skewness

    raw_kurt1_ : array-like, shape (n_features,)
        The raw robust estimated kurtosis

    location_ : array-like, shape (n_features,)
        Estimated robust location (corrected for consistency)

    covariance_ : array-like, shape (n_features, n_features)
        Estimated robust covariance matrix (corrected for consistency)

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix. (stored only if store_precision is True)

    support_ : array-like, shape (n_samples,)
        A mask of the observations that have been used to compute the robust estimates of location and shape.

    dist_ : array-like, shape (n_samples,)
        Mahalanobis distances of the training set (on which `fit` is called) observations.

    raw_skew1_dist_ : array-like, shape (n_samples,)
        Mahalanobis Skewness distances of the training set (on which `fit` is called) observations.

    raw_kurt1_dist_ : array-like, shape (n_samples,)
        Mahalanobis Kurtosis distances of the training set (on which `fit` is called) observations.


    See Also
    --------
    EmpiricalCovariance, MomentMinCovDet

    Notes
    -----
    Outlier detection from covariance estimation may break or not perform well in high-dimensional settings.
    In particular, one will always take care to work with ``n_samples > n_features ** 2``.

    References
    ----------
    ..  [1] Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the minimum
        covariance determinant estimator" Technometrics 41(3), 212 (1999)

    """
    def __init__(self, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None):
        super(MomentAnomalyDetector, self).__init__(store_precision=store_precision, assume_centered=assume_centered, support_fraction=support_fraction, random_state=random_state)
        self.contamination = contamination


    def fit(self, X, y=None):
        """Fit the MomentAnomalyDetector model with X.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : (ignored)
        """
        super(MomentAnomalyDetector, self).fit(X)
        self.threshold_ = sp.stats.scoreatpercentile(self.dist_, 100. * (1. - self.contamination))
        self.prediction_dist_ = None
        return self


    def decision_function(self, X, raw_values=False):
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        raw_values : bool
            Whether or not to consider raw Mahalanobis distances as the
            decision function. Must be False (default) for compatibility
            with the others outlier detection tools.

        Returns
        -------
        decision : array-like, shape (n_samples, )
            Decision function of the samples.
            It is equal to the Mahalanobis distances if `raw_values` is True. By default (``raw_values=False``), it is
            equal to the cubic root of the shifted Mahalanobis distances. In that case, the threshold for being an
            outlier is 0, which ensures a compatibility with other outlier detection tools such as the One-Class SVM.

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        mahal_dist = self.mahalanobis(X)
        if raw_values:
            decision = mahal_dist
        else:
            transformed_mahal_dist = mahal_dist ** 0.33
            decision = self.threshold_ ** 0.33 - transformed_mahal_dist

        return decision


    def predict(self, X):
        """Outlyingness of observations in X according to the fitted robust location and covariance by fast MCD

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        is_outlier : array, shape = (n_samples, ), dtype = bool
            For each observation, tells whether or not it should be considered as an outlier according to the
            fitted model. Values with -1 means outlier and 1 means inlier

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        is_outlier = -np.ones(X.shape[0], dtype=int)
        if self.contamination is not None:
            values = self.decision_function(X, raw_values=True)
            is_outlier[values <= self.threshold_] = 1
        else:
            raise NotImplementedError("You must provide a contamination rate.")

        return is_outlier


    def fitted_mcd_prediction(self, X):
        """Outlyingness of observations in X according to the fitted robust location and covariance by fast MCD
        It is based on self.threshold_, which is a decision function computed during fit.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        is_outlier : array, shape = (n_samples, ), dtype = bool
            For each observation, tells whether or not it should be considered as an outlier according to the
            fitted model. Values with -1 means outlier and 1 means inlier

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        is_outlier = -np.ones(X.shape[0], dtype=int)
        if self.contamination is not None:
            self.prediction_dist_ = self.mahalanobis(X)
            is_outlier[self.prediction_dist_ <= self.threshold_] = 1
        else:
            raise NotImplementedError("You must provide a contamination rate.")

        return is_outlier


    def mcd_prediction(self, X):
        """Outlyingness of observations in X according to the fitted robust location and covariance by fast MCD
        The contamination_threshold is defined by the largest distances between X and the fitted robust location and
        covariance, and the contamination defines the number of largest distances that are predicted as outliers.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        is_outliers : array, shape = (n_samples, ), dtype = bool
            For each observation, tells whether or not it should be considered
            as an outlier according to the fitted model.

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        is_outlier = -np.ones(X.shape[0], dtype=int) # fill with -1 (outliers)
        if self.contamination is not None:
            self.prediction_dist_ = self.mahalanobis(X)
            contamination_threshold = sp.stats.scoreatpercentile(self.prediction_dist_, 100. * (1. - self.contamination))
            is_outlier[self.prediction_dist_ <= contamination_threshold] = 1
        else:
            raise NotImplementedError("You must provide a contamination rate.")

        return is_outlier


    def kurtosis_prediction(self, X):
        """Outlyingness of observations in X according to the fitted robust kurtosis and covariance by fast MCD.
        The contamination_threshold is defined by the largest distances between X and the fitted robust kurtosis and
        covariance, and the contamination defines the number of largest distances that are predicted as outliers.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        pred_label : array, shape = (n_samples, ), dtype = bool
            For each observation, tells whether or not it should be considered as an outlier according, where 0 means
            inliers and 1 means outlisers.

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        pred_label = np.full(X.shape[0], 0, dtype=int)
        if self.contamination is not None:
            # precision
            inv_cov = linalg.pinvh(self.covariance_)

            # kurtosis of the data
            X_kurt1 = X - kurtosis(X, axis=0, fisher=True, bias=True)

            # malhalanobis distance
            mahal_dist = pairwise_distances(X_kurt1, self.raw_kurt1_[np.newaxis, :], metric='mahalanobis', VI=inv_cov)
            mahal_dist = np.reshape(mahal_dist, (len(X_kurt1),)) ** 2  #MD squared
            self.prediction_dist_ = -mahal_dist

            # detect outliers
            contamination_threshold = np.percentile(self.prediction_dist_, 100. * self.contamination)
            pred_label[self.prediction_dist_ <= contamination_threshold] = 1
        else:
            raise NotImplementedError("You must provide a contamination rate.")

        return pred_label


    def skewness_prediction(self, X):
        """Outlyingness of observations in X according to the fitted robust skewness and covariance by fast MCD.
        The contamination_threshold is defined by the largest distances between X and the fitted robust skewness and
        covariance, and the contamination defines the number of largest distances that are predicted as outliers.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        pred_label : array, shape = (n_samples, ), dtype = bool
            For each observation, tells whether or not it should be considered as an outlier according, where 0 means
            inliers and 1 means outlisers.

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        pred_label = np.full(X.shape[0], 0, dtype=int)
        if self.contamination is not None:
            # precision
            inv_cov = linalg.pinvh(self.covariance_)

            # skewness of the data
            X_skew1 = X - skew(X, axis=0, bias=True)

            # malhalanobis distance
            mahal_dist = pairwise_distances(X_skew1, self.raw_skew1_[np.newaxis, :], metric='mahalanobis', VI=inv_cov)
            mahal_dist = np.reshape(mahal_dist, (len(X_skew1),)) ** 2 #MD squared
            self.prediction_dist_ = -mahal_dist

            # detect outliers
            contamination_threshold = np.percentile(self.prediction_dist_, 100. * self.contamination)
            pred_label[self.prediction_dist_ <= contamination_threshold] = 1
        else:
            raise NotImplementedError("You must provide a contamination rate.")

        return pred_label


    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = (n_samples,), optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)