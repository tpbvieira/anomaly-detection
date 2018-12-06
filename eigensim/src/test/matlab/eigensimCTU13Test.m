clear;      % clear variables 
clc;        % clear console
close all;  % close figures

% column_types = {
%     'StartTime': 'str',
%     'Dur': 'single',
%     'Proto': 'uint8',
%     'SrcAddr': 'str',
%     'Sport': 'uint16',
%     'Dir': 'uint8',
%     'DstAddr': 'str',
%     'Dport': 'uint16',
%     'State': 'uint8',
%     'sTos': 'uint8',
%     'dTos': 'uint8',
%     'TotPkts': 'uint16',
%     'TotBytes': 'uint32',
%     'SrcBytes': 'uint32',
%     'Label': 'uint8'}
% drop_features = {
%     'drop_features01': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes', 'Proto'],
%     'drop_features02': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'TotBytes'],
%     'drop_features03': ['SrcAddr', 'DstAddr', 'sTos', 'Sport', 'SrcBytes', 'Proto'],
%     'drop_features04': ['SrcAddr', 'DstAddr', 'sTos', 'Proto']
% }
% opts = detectImportOptions(filePath);
% opts = setvartype(opts,{'Dur'},'single');
% opts = setvartype(opts,{'TotBytes','SrcBytes'},'uint32');
% opts = setvartype(opts,{'Sport','Dport','TotPkts'},'uint16');
% opts = setvartype(opts,{'Proto','Dir','State','sTos','dTos','Label'},'uint8');
% opts = setvartype(opts,{'StartTime','SrcAddr','DstAddr'},'string');

addpath('../../main/matlab/');
addpath('../../main/matlab/eigen_analysis/');
addpath('../../main/matlab/mos/');
addpath('../../main/matlab/rsvd/rsvd');
addpath('../../main/matlab/rsvd/rSVD-single-pass');

resultsFolder = '/home/thiago/dev/projects/discriminative-sensing/mos-eigen-similarity/src/test/matlab/results/';
fileFolder = '/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/csv/';
fileList = dir(fileFolder);

for i = 3:size(fileList,1)
    
    % source
    fileName = fileList(i).name;
    filePath = strcat(fileFolder,fileName);
    
    % destination
    resFolder = strcat(resultsFolder,fileName(1:end-4),'/');
    
    disp('### Loading...')
    X = readtable(filePath);
    % Select data and drop features for features01 set
    X.SrcAddr = [];
    X.DstAddr = [];
    X.sTos = [];
    X.Proto = [];

    X = sortrows(X);
    y = X.Label;
    testFile = strcat(resFolder,'y_test.csv');
    if exist(testFile, 'file') ~= 2
        dlmwrite(testFile,y,'delimiter','\t');
        fprintf('### Saved: %s\n',testFile)
    end
    X.Label = [];
    X.StartTime = [];
    X = table2array(X);
    X = X';

    windowSizes = [20 40 80 160 300];
    windows = [6 10 15 20 40];
    % mos = {'akaike','edc','eft','radoi','sure'};
    mos = {'akaike','edc'};
    ths = [0.1 0.2 0.3 0.4 0.5];

    for wsz = windowSizes
        for ws = windows
            for m = mos
                for th = ths
                    unitStr = sprintf('%d_%d_unit_eig_%s_%0.1f.csv',wsz,ws,m{1},th);
                    destFile = strcat(resFolder, unitStr);
                    if exist(destFile, 'file') ~= 2
                        fprintf('### Test Scenario: features01 - %s\n',destFile)
                        tic;
                        yTest = eigensim(X,wsz,ws,'unit','eig',m{1},th);
                        dlmwrite(destFile,yTest','delimiter','\t');
                        toc;
                    end
                    zmeanStr = sprintf('%d_%d_zmean_eig_%s_%0.1f.csv',wsz,ws,m{1},th);
                    destFile = strcat(resFolder, zmeanStr);
                    if exist(destFile, 'file') ~= 2
                        fprintf('### Test Scenario: features01 - %s\n',destFile)
                        tic;
                        yTest = eigensim(X,wsz,ws,'zmean','eig',m{1},th);
                        dlmwrite(destFile,yTest','delimiter','\t');
                        toc;
                    end
                end
            end
        end
    end
end