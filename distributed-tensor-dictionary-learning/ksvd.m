function [A_hat, S_hat] = ksvd(noIt, X, A_hat, solver, varargin)
    % ...
    %
    % This code is a version of Karl Skretting work.
    %----------------------------------------------------------------------
    % parameters:
    %   noIt        = number of iterations
    %   K           = number of observations
    %   X           = data matrix (training data)
    %   Dict        = dictionary
    %   solver      = selected solver, see sparseapprox function
    %   varargin    = additional arguments, see sparseapprox function
    %---------------------------------------------------------------------- 
    K = size(A_hat,2);    
    for it = 1:noIt
        S_hat = sparseapprox(X, A_hat, solver, varargin);
        R = X - A_hat * S_hat; 
        for k=1:K
            I = find(S_hat(k,:));
            if ~isempty(I)
                Ri = R(:,I) + A_hat(:,k) * S_hat(k,I);
                [U,S,V] = svds(Ri,1,'L');
                A_hat(:,k) = U;
                S_hat(k,I) = S * V';
                R(:,I) = Ri - A_hat(:,k) * S_hat(k,I);
            end
        end
    end