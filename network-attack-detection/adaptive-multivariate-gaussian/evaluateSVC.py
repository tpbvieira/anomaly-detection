import os
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import classification_report

def main():
    directory = os.fsencode(os.path.join('/media/thiago/ubuntu/datasets/network/','stratosphere-botnet-2011/CTU-13-Dataset/','ctu13pickles/'))

    for ctu13pickle in os.listdir(directory):
        strPath = os.path.join(directory.decode('utf-8'), ctu13pickle.decode('utf-8'));
        print('Processing ',strPath)

        # read in the data set and prep it for training
        df = pd.read_pickle(strPath)
        df.dropna(inplace=True)
        y = df['label']
        df.drop(['label'], axis=1, inplace=True)
        print(df.shape)
        
        # split into train and test data
        X_train, X_test, y_train, y_test = tts(df, y, train_size=0.7)
        print("Dataframe: shape {}".format(df.shape))
        print(df.head())
        print("Training SVM")

        # train SVC classifier
        clf = SVC(verbose=True)
        clf.fit(X_train, y_train)
        print(clf)

        # Predict
        p = clf.predict(X_test)
        report = classification_report(y_test, p)
        print(report)

if __name__ == '__main__': main()