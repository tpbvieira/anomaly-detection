clc;
clear all;
close all;

addpath('/home/thiago/Dropbox/dev/matlab/mos/');

covGETV = [1887545 2341327 3213867 133238294 92384021611 708335];
corGETV = [2.0734 2.1451 10.0718 2.1620 2.4253 1.7948];
periodsSize = 20;

%AIC (1)
fprintf('Cov_akaike: %i\n',akaike_short2(covGETV, periodsSize));
fprintf('Cor_akaike: %i\n',akaike_short2(corGETV, periodsSize));

%MDL (2)
fprintf('Cov_mdl: %i\n',mdl_short2(covGETV,periodsSize));
fprintf('Cor_mdl: %i\n',mdl_short2(corGETV,periodsSize));

%EDC (3)
fprintf('Cov_edc: %i\n',edc_short2(covGETV,periodsSize));
fprintf('Cor_edc: %i\n',edc_short2(corGETV,periodsSize));

%RADOI (4)
fprintf('Cov_ranoi: %i\n',ranoi_app(covGETV));
fprintf('Cor_ranoi: %i\n',ranoi_app(corGETV));

% %EFT (5)
% c = struct2cell(load('../data/Pfa'));
% Pfa = c{1};
% c = struct2cell(load('../data/coeff'));
% coeff = c{1};
% c = struct2cell(load('../data/q'));
% waiting = c{1};
% getv_size = size(getv);
% %[eft_coeff,prob_found] = calc_coef_paper(getv_size(2),periodsSize,Pfa,coeff,waiting);
% eft_coeff = [0.3000    0.4000    0.4000    0.4000    0.4000];
% eft_short(getv,eft_coeff,getv_size(2),periodsSize)

% %SURE (6)
% sureM = NaN;
% sure_method(sureM,numberOfPorts,periodsSize)