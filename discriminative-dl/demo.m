close all; clear all; clc;

addpath([cd '/utilies']);
load(['AR_EigenFace']);

tic;
%% FDDL parameter
fprintf('[Demo] Trainning Fisher Discriminative Dictionaries from train data...\n');
opts.nClass = 5;
opts.wayInit = 'PCA';
opts.lambda1 = 0.005;
opts.lambda2 = 0.05;
opts.nIter = 15;
opts.show = false;
[Dict, Drls, CoefM, CMlabel] = FDDL(tr_dat, trls, opts);
toc;

tic;
%% Sparse classification of test data
fprintf('\n[Demo] Sparse classification of test data...\n');
lambda = 0.005;
nClass = opts.nClass;
weight = 0.5;
td1_ipts.D = Dict;
td1_ipts.tau1 = lambda;
if size(td1_ipts.D, 1) >= size(td1_ipts.D, 2)
    td1_par.eigenv = eigs(td1_ipts.D' * td1_ipts.D, 1);
else
    td1_par.eigenv = eigs(td1_ipts.D * td1_ipts.D', 1);
end
ID = [];

fprintf(['[Demo] Sparse Coding ' num2str(size(tt_dat, 2)) ' tests and classification according to the lowest (gap + weight * gCoef3)\n']);
for indTest = 1:size(tt_dat, 2)
    % sparse coding
    td1_ipts.y = tt_dat(:, indTest);                                        % get test data of indTest
    [opts] = IPM_SC(td1_ipts, td1_par);                                     % sparse code the test data of indTest into s
    s = opts.x;                                                             % s is the sparse coding of indTest data test for all classes
    
    % compute the gap and coefficient of s for each class
    for indClass = 1:nClass
        temp_s = zeros(size(s));                                            % init temp_s
        temp_s(indClass==Drls) = s(indClass==Drls);                         % fill temp_s with the sparse code of indClass according to learnt labels Drls, but the remaining values are zero
        zz = tt_dat(:, indTest) - td1_ipts.D * temp_s;                      % get error after recovery
        gap(indClass) = zz(:)' * zz(:);                                     % comput the gap of indClass
        mean_coef_c = CoefM(:, indClass);                                   % get the learnt mean coefficient
        gCoef3(indClass) = norm(s - mean_coef_c, 2) ^ 2;                      % normalize and store the coefficient of indClass
    end
    
    % the lowest (gap + weight * gCoef3) indicates the corresponding label
    wgap3 = gap + weight * gCoef3;
    index3 = find(wgap3 == min(wgap3));
    id3 = index3(1);
    ID = [ID id3];
end
toc;

fprintf('[Demo] %s%8f%s%8f\n', 'reco_rate = ', sum(ID==ttls)/(length(ttls)), ' of ', (length(ttls)));