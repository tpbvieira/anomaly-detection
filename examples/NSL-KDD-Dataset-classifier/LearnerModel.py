from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class LearnerModel:
    """
    This module defines a class for the learner module.
    """

    def __init__(self, train_x, train_y, test_x, test_y):
        """
        :param train_x: Trainer data set of features
        :param train_y: Trainer data set of labels.
        :param test_x: Test data set of features.
        :param test_y: Test data set of labels.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def perform_variance_threshold(self, v_threshold):
        selector = VarianceThreshold(v_threshold)
        self.train_x = selector.fit_transform(self.train_x, self.train_y)

        self.test_x = selector.transform(self.test_x)

    def perform_standard_scalar_fit_transform_predict(self):
        """
        Performs StandardScalar.fit() transform() and predict()
        """
        sc = StandardScaler()
        sc.fit(self.train_x)

        print 'Performing transform'
        x_train_std = sc.transform(self.train_x)
        x_test_std = sc.transform(self.test_x)

        print 'Performing Fit'
        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(x_train_std, self.train_y)

        print 'Creating Predictor model'
        y_pred = svm.predict(x_test_std)

        print ('Misclassified samples: %d' % (self.test_y != y_pred).sum())
        print ('Accuracy: %.2f %%' % (accuracy_score(self.test_y, y_pred) * 100))

    def perform_knn_classification(self):
        knn = KNeighborsClassifier()
        print 'KNN: Performing Fit'
        knn.fit(self.train_x, self.train_y)

        print 'KNN: Beginning Predict'
        y_pred = knn.predict(self.test_x)

        print ('Misclassified samples: %d' % (self.test_y != y_pred).sum())
        print ('Accuracy: %.2f %%' % (accuracy_score(self.test_y, y_pred) * 100))

    def perform_decision_tree_classification(self):
        """Decision Tree classifier"""
        dtc = DecisionTreeClassifier()

        print 'DecisionTreeClassifier: Performing Fit'
        dtc.fit(self.train_x, self.train_y)

        print 'DecisionTreeClassifier: Beginning Predict'
        y_pred = dtc.predict(self.test_x)

        print ('Misclassified samples: %d' % (self.test_y != y_pred).sum())
        print ('Accuracy: %.2f %%' % (accuracy_score(self.test_y, y_pred) * 100))
