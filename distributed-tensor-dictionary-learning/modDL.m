function [A_hat,S_hat] = modDL(noIt, X, A_hat, solver, varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Method of Optimized Directions (MOD)                            %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Author: Thiago Vieira, UnB, 2017                                %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % General goal: ...
    %
    % For details, please see:
    % TENSOR-BASED ALGORITHMS FOR LEARNING MULTIDIMENSIONAL SEPARABLE 
    % DICTIONARIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input: 
    %    noIt       = maximum number of iterations
    %    X          = M x T training data.
    %    A_hat      = initial guess for the M x N dictionary
    %    solver     = selected solver
    %    varargi    = additional parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    for it = 1:noIt
        S_hat = sparseapprox(X, A_hat, solver, varargin);
        A_hat = X * pinv(S_hat);
        A_hat = dictnormalize(A_hat);
    end