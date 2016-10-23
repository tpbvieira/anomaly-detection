# dataframe.py
# This is the dataframe code.
# This is a pillar code, i.e these are the helper codes.

# This script will create data and target as a numpy array
# and it can be used by others.

# ------------------------------------------------------------------------------
# This python script will read the provided dataset
# and it will generate a list 'df_data' and 'df_target' which are numpy arrays.

import os
import numpy as np
import values


def __get_all_lines_from_dataset(dataset_file):
    """Opens the dataset_file in READ MODE; returns the lines of the file."""
    with open(dataset_file) as f:
        lines = f.readlines()

    return lines


def __convert_feature_to_float(features):
    """Converts the string attributes(Actual features) to appropriate float values"""

    temp_arr = []
    pair_values = values.get_list()

    for attrib in features:
        # Try to convert the attrib to float,
        # If that is not possible then look up the pair_values[]
        try:
            temp_arr.append(float(attrib))
        except ValueError:
            temp_arr.append(pair_values[attrib])

    return temp_arr


def get_data_set(data_set_file, absolute_path=False, test_data_only=False):
    """Read the dataset file and then return it as NumPy array.
    Does conversion from string to float by looking up the pair_value list.
    :param data_set_file: File containing data set.
    :param absolute_path: if True, then data_set_file is treated to be a path; if False then the default path
    (C:\users\user\documents\dataset) is taken as the directory
    :param test_data_only: if True then only the features will be returned.
    :return Numpy array of features and optional label.
    """

    if absolute_path:
        file_name = data_set_file
    else:
        file_name = '%s\\Documents\\dataset\\%s' % (os.path.expanduser('~'), data_set_file)

    debug = True

    dataset_lines = __get_all_lines_from_dataset(file_name)

    if debug:
        print 'Stats of %s' % file_name
        print 'Has %d instances, each instance with %d attributes' % (
            len(dataset_lines), len(dataset_lines[0].split(',')))

    # Get the key value pair.
    pair_values = values.get_list()

    features = []
    label = []  # contains the target label.

    for line in dataset_lines:
        items = line.replace('\n', '').split(',')

        # Get the n-1 data, (i.e features)
        t_features = items[:len(items) - 1]

        temp_arr = __convert_feature_to_float(t_features)

        features.append(temp_arr)

        # append the label to target list.
        label.append(pair_values[items[-1]])

    # convert the regular list into numpy array.
    df_data = np.asarray(features)
    df_target = np.asarray(label)

    if test_data_only:
        return df_data
    else:
        return df_data, df_target
