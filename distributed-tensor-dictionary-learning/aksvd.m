function Dict = aksvd(noIt, K, X, Dict, solver, varargin)
    % ...
    %
    % This code is a version of Karl Skretting work.
    %-------------------------------------------------------------------------
    % parameters:
    %   noIt        = number of iterations
    %   K           = number of observations
    %   X           = data matrix
    %   Dict        = dictionary
    %   solver      = selected solver, see sparseapprox function
    %   varargin    = additional arguments, see sparseapprox function
    %-------------------------------------------------------------------------    

    for it = 1:noIt
        W = sparseapprox(X, Dict, solver, varargin);                        % find weights, using dictionary D
        R = X - Dict * W;
        for a = 1:3
            for k=1:K
                I = find(W(k,:));
                Ri = R(:,I) + Dict(:,k)*W(k,I);
                dk = Ri * W(k,I)';
                dk = dk/sqrt(dk'*dk);                                       % normalize
                Dict(:,k) = dk;
                W(k,I) = dk'*Ri;
                R(:,I) = Ri - Dict(:,k)*W(k,I);
            end
        end        
    end
    
    return