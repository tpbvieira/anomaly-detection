clc; close all;
% for testing, execute:
% result = runtests('mosTest');
% rt = table(result)

% initiate variables
addpath('/home/thiago/dev/projects/discriminative-sensing/mos-eigen-similarity/src/main/matlab/mos/');
covGETV = [1887545 2341327 3213867 133238294 92384021611 708335];
corGETV = [2.0734 2.1451 10.0718 2.1620 2.4253 1.7948];
periodsSize = 20;

%% AIC
cov_akaike = akaike_short2(covGETV, periodsSize);
cor_akaike = akaike_short2(corGETV, periodsSize);
assert(cov_akaike == 2)
assert(cor_akaike == 1)

%% MDL
cov_mdl = mdl_short2(covGETV,periodsSize);
cor_mdl = mdl_short2(corGETV,periodsSize);
assert(cov_mdl == 2)
assert(cor_mdl == 1)

%% EDC
cov_edc = edc_short2(covGETV,periodsSize);
cor_edc = edc_short2(corGETV,periodsSize);
assert(cov_edc == 2)
assert(cor_edc == 1)

%% RADOI
cov_ranoi = ranoi_app(covGETV);
cor_ranoi = ranoi_app(corGETV);
assert(cov_ranoi == 5)
assert(cor_ranoi == 1)

%% SURE
% sureM = NaN;
% sure_method(sureM,numberOfPorts,periodsSize)

%% EFT
% c = struct2cell(load('data/Pfa'));
% Pfa = c{1};
% c = struct2cell(load('data/coeff'));
% coeff = c{1};
% c = struct2cell(load('data/q'));
% waiting = c{1};
% getv_size = size(getv);
% [eft_coeff, prob_found] = calc_coef_paper(getv_size(2), periodsSize, Pfa, coeff, waiting)
% eft_coeff = [0.3000    0.4000    0.4000    0.4000    0.4000];
% eft_short(getv,eft_coeff,getv_size(2),periodsSize)

