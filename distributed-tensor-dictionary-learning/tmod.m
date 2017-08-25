function [A_hat,S_hat] = tmod(noIt, X, A_hat1, A_hat2, solver, varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% tensor-based Method of Optimized Directions (MOD)                   %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Author: Thiago Vieira, UnB, 2017                                    %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % General goal: ...
    %
    % For details, please see:
    % TENSOR-BASED ALGORITHMS FOR LEARNING MULTIDIMENSIONAL SEPARABLE DICTIONARIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input: 
    %    noIt       = maximum number of iterations
    %    X          = M x T training data.
    %    Dict       = initial guess for the M x N dictionary
    %    solver     = selected solver
    %    varargin	= additional parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [M1,N1] = size(A_hat1);
    [M2,N2] = size(A_hat2);
    [M,T] = size(X);
    Xt = permute(reshape(X.',[T,M2,M1]),[3,2,1]);    
    Xt1 = reshape(permute(Xt,[1,3,2]),M1,[]);
    Xt2 = reshape(permute(Xt,[2,1,3]),M2,[]);
    I_t = eye(T, 'single');
    for it = 1:noIt
        S_hat = sparseapprox(X, kron(A_hat1,A_hat2), solver, varargin);
        
        St_hat = permute(reshape(S_hat.', [T,N2,N1]), [3,2,1]);
        St_hat1 = reshape(permute(St_hat,[1,3,2]),N1,[]);
        St_hat2 = reshape(permute(St_hat,[2,1,3]),N2,[]);
                
        A_hat1 = Xt1 * pinv(St_hat1 * (kron(A_hat2, I_t)).');
        A_hat2 = Xt2 * pinv(St_hat2 * (kron(I_t, A_hat1)).');
        
        A_hat1 = dictnormalize(A_hat1);
        A_hat2 = dictnormalize(A_hat2);
    end
    A_hat = kron(A_hat1,A_hat2);