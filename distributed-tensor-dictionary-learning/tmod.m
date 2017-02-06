function A_hat = tmod(noIt, X, A_hat, A_hat1, A_hat2, solver, varargin)
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
    X_t = permute(reshape(X,[T,M2,M1]),[3,2,1]);
    X_t1 = reshape(X_t,[M1,M2*T]);
    X_t2 = reshape(permute(X_t,[2,1,3]),[M2,M1*T]);
    for it = 1:noIt
        S_hat = sparseapprox(X, kron(A_hat1,A_hat2), solver, varargin);
        
        S_hat_t = permute(reshape(S_hat,[T,N2,N1]),[3,2,1]);
        S_hat_t1 = reshape(S_hat_t,[N1,N2*T]);
        S_hat_t2 = reshape(permute(S_hat_t,[2,1,3]),[N2,N1*T]);
                
        A_hat1 = X_t1 * pinv(S_hat_t1 * (kron(A_hat2,noIt)).');
        A_hat2 = X_t2 * pinv(S_hat_t2 * (kron(noIt,A_hat1)).');
        
        A_hat1 = dictnormalize(A_hat1);
        A_hat2 = dictnormalize(A_hat2);
    end    
    return;