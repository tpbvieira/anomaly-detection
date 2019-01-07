# coding=utf-8
If you really want to use neighbors.LocalOutlierFactor for novelty detection, i.e. predict labels or compute the score of abnormality of new unseen data, you can instantiate the estimator with the novelty parameter set to True before fitting the estimator. In this case, fit_predict is not available.
Warning Novelty detection with Local Outlier Factor
When novelty is set to True be aware that you must only use predict, decision_function and score_samples on new unseen data and not on the training samples as this would lead to wrong results. The scores of abnormality of the training samples are always accessible through the negative_outlier_factor_ attribute.
The behavior of neighbors.LocalOutlierFactor is summarized in the following table.
Method	Outlier detection	Novelty detection
fit_predict	OK	Not available
predict	Not available	Use only on new data
decision_function	Not available	Use only on new data
score_samples	Use negative_outlier_factor_	Use only on new data