clear; % clear variables
clc; % clear console
close all; % close figures

addpath('../../main/matlab/');
addpath('../../main/matlab/eigen_analysis/');
addpath('../../main/matlab/mos/');
addpath('../../main/matlab/rsvd/rsvd');
addpath('../../main/matlab/rsvd/rSVD-single-pass');

% data
tic;
disp('### Loading...')
X = dlmread('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/drop_features01/train.csv', '\t');
X = X';
X = single(X);
% y = dlmread('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/drop_features01/train_label.csv', '\t');
toc;

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
                if exist(unitStr, 'file') ~= 2
                    % Test Scenario 1 - 20_6_unit_eig_edc_0.30
                    fprintf('### Test Scenario features01 - %s\n',unitStr)
                    tic;
                    yTest = eigensim(X,wsz,ws,'unit','eig',m{1},th);
                    dlmwrite(unitStr,yTest,'delimiter','\t');
                    toc;
                end
                zmeanStr = sprintf('%d_%d_zmean_eig_%s_%0.1f.csv',wsz,ws,m{1},th);               
                if exist(zmeanStr, 'file') ~= 2
                    % Test Scenario 1 - 20_6_zmean_eig_edc_0.30
                    tic;
                    fprintf('### Test Scenario features01 - %s\n',zmeanStr)
                    yTest = eigensim(X,wsz,ws,'zmean','eig',m{1},th);
                    dlmwrite(zmeanStr,yTest,'delimiter','\t');
                    toc;
                end
            end
        end
    end
end