clc; clear all; close all;

X1 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/1.txt', '\t');
X2 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/2.txt', '\t');
X3 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/3.txt', '\t');
X4 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/4.txt', '\t');
X5 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/5.txt', '\t');
X6 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/6.txt', '\t');
X = [X1,X2,X3,X4,X5,X6];

eigensim(X,20,6,'zmean','edc', 0.40)