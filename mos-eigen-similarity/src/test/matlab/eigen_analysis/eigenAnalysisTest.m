clc;clear all;close all;

addpath('../../../main/matlab/eigen_analysis/');
X = dlmread(['/media/thiago/shared/backup/doutorado/data/all/traffic/1.txt'], '\t');

%% eigencorrelationTest
[Sc,Ec,Vrc,Mrc] = eigencorrelation(X);

%% eigencovarianceTest
[Sc,Ec,Vrc,Mrc] = eigencovariance(X);