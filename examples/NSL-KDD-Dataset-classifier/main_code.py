# main_code.py
# This is the entry point of the program.

import tkFileDialog
from Tkinter import Tk

import dataframe
from LearnerModel import LearnerModel

STATS_FOR_NERDS = False

# def perform_train_test_split():
#     x, y = dataframe.get_dataset_from_file('nsl.train')
#     test_x, test_y = dataframe.get_dataset_from_file('nsl.test')
#
#     v_threshold = 0.15
#
#     new_x = VarianceThresholdTest.get_transformed_matrix_with_threshold(x, y, v_threshold)
#
#     print 'After VarianceThreshold data contains %d features' % (len(new_x[0]))
#
#     split_ratio = 0.1
#
#     while split_ratio < 1.0:
#         X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=split_ratio, random_state=0)
#
#         print 'Stats for nerds'
#         print len(X_train), len(X_test), len(y_train), len(y_test)
#         print len(X_train[0]), len(X_test[0])
#
#         print 'Calling StandardScalar with split_ratio = %f' % split_ratio
#
#         sc = StandardScaler()
#         sc.fit(X_train)
#
#         print 'Done with StandardScalar fit'
#
#         X_train_std = sc.transform(X_train)
#         X_test_std = sc.transform(X_test)
#
#         print 'Calling SVC'
#         svm = SVC(kernel='linear', C=1.0, random_state=0)
#         svm.fit(X_train_std, y_train)
#
#         print 'Done with svm fit'
#
#         print 'Beginning Predict...'
#         y_pred = svm.predict(X_test_std)
#         print 'Done with predict.'
#
#         print ('Misclassified samples: %d' % (y_test != y_pred).sum())
#
#         print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred) * 100))
#         print '-' * 80
#
#         split_ratio += 0.1

# This is no longer needed.
# def perform_svm_test():
#     v_threshold = 0.05
#     while v_threshold < 1.0:
#         sel = VarianceThreshold(v_threshold)
#         x = sel.fit_transform(x)
#         test_x = sel.transform(test_x)
#
#         print 'Support indices', sel.get_support(indices=True)
#         print 'x size {0}, test_x size {1}'.format(len(x[0]), len(test_x[0]))
#
#         # X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=split_ratio, random_state=0)
#
#         sc = StandardScaler()
#         sc.fit(x)
#
#         print 'Done with StandardScalar fit'
#
#         X_train_std = sc.transform(x)
#         X_test_std = sc.transform(test_x)
#
#         print 'Calling SVC'
#         svm = SVC(kernel='linear', C=1.0, random_state=0)
#         svm.fit(X_train_std, y)
#         print 'Done with svm fit'
#
#         print 'Beginning Predict...'
#         y_pred = svm.predict(X_test_std)
#         print 'Done with predict.'
#
#         print y_pred[0], y[0], test_y[0]
#
#         # print ('Misclassified samples: %d' % (test_y != y_pred).sum())
#         error_count = 0
#
#         # if len(y_pred) == len(test_y):
#         #     l = len(y_pred)
#         #
#         #     for i in range(l):
#         #         if y_pred[i] != test_y[i]:
#         #             error_count += 1
#
#         error_count = ((y_pred != test_y).sum())
#         success_count = len(test_y) - error_count
#
#         print 'Error count', error_count
#         print 'Success count', success_count
#
#         print 'Accuracy ratio of Success: %f' % ((float(success_count) / len(test_y)) * 100)
#
#         # print ('Accuracy: %.2f' % (accuracy_score(test_y, y_pred) * 100))
#         print '-' * 80
#
#         v_threshold += 0.05
#
#         x, y = dataframe.get_dataset_from_file('nsl.train')
#         test_x, test_y = dataframe.get_dataset_from_file('nsl.test')


# all_test_datasets = glob.glob('C:\Users\Preetham\Documents\dataset\\nsl_independent\*')
# print 'Current datasets are', all_test_datasets

if __name__ == '__main__':
    # perform_train_test_split()

    Tk().withdraw()

    train_data_set = tkFileDialog.askopenfilename()
    test_data_set = tkFileDialog.askopenfilename()

    if train_data_set != "" and test_data_set != "":
        print "Chosen data sets are: ", train_data_set, test_data_set

        x, y = dataframe.get_data_set(train_data_set, absolute_path=True)

        test_x, test_y = dataframe.get_data_set(test_data_set, absolute_path=True)

        # Creating LearnerModel is Compulsory.
        learner = LearnerModel(x, y, test_x, test_y)

        learner.perform_knn_classification()
        print '*' * 80

        learner.perform_decision_tree_classification()
        print '*' * 80

        learner.perform_variance_threshold(0.15)
        learner.perform_standard_scalar_fit_transform_predict()
    else:
        print 'Please specify data set'
