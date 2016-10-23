clc;
clear all;
close all;

X = dlmread(['/media/thiago/shared/backup/doutorado/data/all/traffic/1.txt'], '\t');
[Sc,Ec,Vrc,Mrc] = eigencorrelation(X);
[Sc,Ec,Vrc,Mrc] = eigencovariance(X)