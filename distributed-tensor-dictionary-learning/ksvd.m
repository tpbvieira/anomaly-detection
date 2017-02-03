function Dict = ksvd(noIt, K, X, Dict, solver, varargin)
    % ...
    %
    % This code is a version of Karl Skretting work.
    %-------------------------------------------------------------------------
    % parameters:
    %   noIt        = number of iterations
    %   K           = number of observations
    %   X           = data matrix (training data)
    %   Dict        = dictionary
    %   solver      = selected solver, see sparseapprox function
    %   varargin    = additional arguments, see sparseapprox function
    %------------------------------------------------------------------------- 
    
    for it = 1:noIt
        W = sparseapprox(X, Dict, solver, varargin);                    % find weights, using dictionary D
        R = X - Dict*W; 
        for k=1:K
            I = find(W(k,:));
            Ri = R(:,I) + Dict(:,k)*W(k,I);
            [U,S,V] = svds(Ri,1,'L');
            Dict(:,k) = U;
            W(k,I) = S*V';
            R(:,I) = Ri - Dict(:,k)*W(k,I);
        end
    end
    
    return