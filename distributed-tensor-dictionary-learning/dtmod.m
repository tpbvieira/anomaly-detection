    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% distributed tensor-based Method of Optimized Directions (MOD)   %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Author: Thiago Vieira, UnB, 2017                                %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % General goal: ...
    %
    % we can have a model of the form S x1 A1 x2 A2 x3 A3 if we have 
    % a 3-fold separable manifold, however, we then need S to be 4-way 
    % (N1 x N2 x N3 x T).
    % 
    % The reason is that the underlying matrix-valued model is always 
    % Y = A*S, since columns of Y are sparsely represented by S in the 
    % dictionary A. So far we've used A = A1 \kron A2 but we might as well 
    % consider A = A1 \kron A2 \kron ... \kron AR, leading to an (R+1)-way 
    % Tucker with R free factors.
    % 
    % If you wanted a "full" Tucker we would have to factorize the sparse 
    % core tensor S (which is N1 x N2 x T) into some Stilde x3 C. But I'm 
    % not seeing this as S has no low-rank structure, its only structure 
    % is that it is sparse. Since we want to sparsity patterns to vary 
    % somewhat randomly over the T observations, I think that imposing a 
    % low-rank structure over time would be too stringent. But maybe I'm 
    % overlooking something also.
    % 
    % For details, please see:
    % ???
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input: 
    %    noIt       = maximum number of iterations
    %    X          = M x T training data.
    %    Dict       = initial guess for the M x N dictionary
    %    solver     = selected solver
    %    varargin	= additional parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clc;
    clf;
    clear all;

    s = 5;
    snr = 20;
    L = 500;
    nofTrials = 1;
    noIt = 5;
    I = 8;
    J = 6;
    K = 2;
    
    M1 = 4;
    M2 = 2;
    N1 = 4;
    N2 = 3;
    
    % data generation
    [A, A1, A2] = makedict2(M1, M2, N1, N2, 'G');
    X = datamake(A, L, s, snr, 'G');
    A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );         % Normalize and arrange the vectors for an initial estimated dictionary        
    [A_hat1, A_hat2] = krondecomp(A_hat, M1, M2, N1, N2);               % Make the data separable decomposing the approximation of A_hat and generating new A_hat*
    T = size(X,2);
    
    Xt = permute(reshape(X.',[T,M2,M1]),[3,2,1]);    
    Xt1 = reshape(permute(Xt,[1,3,2]),M1,[]);
    Xt2 = reshape(permute(Xt,[2,1,3]),M2,[]);
    I_t = eye(T);
    
    %% minimize
    for it = 1:noIt
        A_hat = kron(A_hat1, A_hat2);
        S_hat = sparseapprox(X, A_hat, 'javaMP', 'tnz',s);
        size(S_hat)
        
        St_hat = permute(reshape(S_hat.', [T,N2,N1]), [3,2,1]);
        St_hat1 = reshape(permute(St_hat,[1,3,2]),N1,[]);
        St_hat2 = reshape(permute(St_hat,[2,1,3]),N2,[]);
                
        A_hat1 = Xt1 * pinv(St_hat1 * (kron(A_hat2,I_t)).');
        A_hat2 = Xt2 * pinv(St_hat2 * (kron(I_t,A_hat1)).');
        
        A_hat1 = dictnormalize(A_hat1);
        A_hat2 = dictnormalize(A_hat2);
    end
    
    %% minimized dictionary
    A_hat = kron(A_hat1,A_hat2);
    return;