function [Dict, Drls, CoefM, CMlabel] = FDDL(TrainDat, TrainLabel, opts)
% ========================================================================
% Fisher Discriminative Dictionary Learning (FDDL), Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for learning the
% Fisher Discriminative Dictionary from a labeled training data
%
% Please refer to the following paper
%
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang, "Fisher Discrimination Dictionary Learning for Sparse Representation", In IEEE Int. Conf. on Computer Vision, 2011.
%----------------------------------------------------------------------
% Input : (1) TrainDat: the training data matrix.
%                       Each column is a training sample
%         (2) TrainDabel: the training data labels
%         (3) opts      : the struture of parameters
%               .nClass   the number of classes
%               .wayInit  the way to initialize the dictionary
%               .lambda1  the parameter of l1-norm energy of coefficient
%               .lambda2  the parameter of l2-norm of Fisher Discriminative
%               coefficient term
%               .nIter    the number of FDDL's iteration
%               .show     sign value of showing the gap sequence
% Output:
%   (1) Dict: the learnt dictionary via FDDL
%   (2) Drls: the labels of learnt dictionary's columns
%   (2) CoefM: Mean Coefficient Matrix. Each column is a mean coef vector
%   (3) CMlabel: the labels of CoefM's columns.
%-----------------------------------------------------------------------
%
% Usage:
% Given a training data, including TrainDat and TrainLabel, and the
% parameters, opts.
%
% [Dict, CoefM, CMlabel] = FDDL(TrainDat, TrainLabel, opts)
%-----------------------------------------------------------------------

% normalize energy
TrainDat = TrainDat * diag(1./sqrt(sum(TrainDat.*TrainDat)));

% initialize dicttionary and label of each class, appending dicttionary and
% label for each new class
Dict_ini = [];
Dlabel_ini = [];
for ci = 1:opts.nClass
    ci_data = TrainDat(:, TrainLabel==ci);
    ci_dict = FDDL_INID(ci_data, size(ci_data, 2), opts.wayInit);
    Dict_ini = [Dict_ini ci_dict];
    Dlabel_ini = [Dlabel_ini repmat(ci, [1 size(ci_dict, 2)])];
end

% initialize coef without between-class scatter
ini_par.tau = opts.lambda1;
ini_par.lambda = opts.lambda2;
ini_ipts.D = Dict_ini;
coef = zeros(size(Dict_ini, 2), size(TrainDat, 2));
if size(Dict_ini, 1) > size(Dict_ini, 2)
    ini_par.c = 1.05 * eigs(Dict_ini' * Dict_ini, 1);
else
    ini_par.c = 1.05 * eigs(Dict_ini * Dict_ini', 1);
end

% Coefficient Initialization by FDDL_INIC for each class,
fprintf(['[FDDL] Initializing Coef for ' num2str(opts.nClass) ' classes\n']);
for ci = 1:opts.nClass
    ini_ipts.X = TrainDat(:, TrainLabel==ci);                               % X is the train dat of ci
    [ini_opts] = FDDL_INIC(ini_ipts, ini_par);                              % A = the coefficient matrix and ert = total energy sequence
    coef(:, TrainLabel == ci) = ini_opts.A;                                 % update coefficient of ci to A
end

% Fisher Discriminative Dictionary Learning
Fish_par.dls = Dlabel_ini;
Fish_ipts.D = Dict_ini;
Fish_ipts.trls = TrainLabel;
Fish_par.tau = opts.lambda1;
Fish_par.lambda2 = opts.lambda2;
Fish_nit = 1;
drls = Dlabel_ini;
fprintf(['[FDDL] Learning Fisher Discriminative Dictionary, ' num2str(opts.nIter) ' iterations\n'])
while Fish_nit <= opts.nIter
    
    if size(Fish_ipts.D, 1) > size(Fish_ipts.D, 2)
        Fish_par.c = 1.05 * eigs(Fish_ipts.D' * Fish_ipts.D, 1);
    else
        Fish_par.c = 1.05 * eigs(Fish_ipts.D * Fish_ipts.D', 1);
    end
    
    % updating the coefficient for each class
    fprintf(['[FDDL] It: ' num2str(Fish_nit) ' - Updating coefficients of ' num2str(opts.nClass) ' classes\n'])
    for ci = 1:opts.nClass
        Fish_ipts.X = TrainDat(:, TrainLabel==ci);                          % get train data of ci
        Fish_ipts.A = coef;                                                 % init coef
        Fish_par.index = ci;                                                % ci index
        [Copts] = FDDL_SpaCoef(Fish_ipts, Fish_par);                        % Coefficient updating of FDDL, A = the coefficient matrix and ert = total energy sequence
        coef(:, TrainLabel==ci) = Copts.A;
        CMlabel(ci) = ci;
        CoefM(:, ci) = mean(Copts.A, 2);
    end
    [GAP_coding(Fish_nit)] = FDDL_FDL_Energy(TrainDat, coef, opts.nClass, Fish_par, Fish_ipts);
    
    % updating the dictionary for each class
    fprintf(['[FDDL] It: ' num2str(Fish_nit) ' - Updating dictionary of ' num2str(opts.nClass) ' classes\n']);
    for ci = 1:opts.nClass
        [Fish_ipts.D(:, drls==ci), Delt(ci).delet] = FDDL_UpdateDi(TrainDat, coef, ci, opts.nClass, Fish_ipts, Fish_par);
    end
    [GAP_dict(Fish_nit)] = FDDL_FDL_Energy(TrainDat, coef, opts.nClass, Fish_par, Fish_ipts);
    
    % updating dictionary, label and coefficient
    newD = []; newdrls = []; newcoef = [];
    for ci = 1:opts.nClass
        delet = Delt(ci).delet;
        if isempty(delet)
            classD = Fish_ipts.D(:, drls==ci);
            newD = [newD classD];
            newdrls = [newdrls repmat(ci, [1 size(classD, 2)])];
            newcoef = [newcoef; coef(drls==ci, :)];
        else
            temp = Fish_ipts.D(:, drls==ci);
            temp_coef = coef(drls==ci, :);
            for temp_i = 1:size(temp, 2)
                if sum(delet==temp_i)==0
                    newD = [newD temp(:, temp_i)];
                    newdrls = [newdrls ci];
                    newcoef = [newcoef;temp_coef(temp_i, :)];
                end
            end
        end
    end
    
    Fish_ipts.D = newD;
    coef = newcoef;
    drls = newdrls;
    Fish_par.dls = drls;
    Fish_nit = Fish_nit +1;
end

Dict = Fish_ipts.D;
Drls = drls;

if opts.show
    subplot(1, 2, 1);
    plot(GAP_coding, '-*');
    title('GAP Coding');
    
    subplot(1, 2, 2);
    plot(GAP_dict, '-o');
    title('GAP Dictionary');
end

return;