function Dict = ilsdlaMatlab(noIt, X, Dict, solve, varargin)
    % ...
    %
    % This code is a version of Karl Skretting work.
    %-------------------------------------------------------------------------
    % parameters:
    %   noIt        = number of iterations
    %   X           = data matrix
    %   Dict        = dictionary
    %   solver      = selected solver, see sparseapprox function
    %   varargin    = additional arguments, see sparseapprox function
    %------------------------------------------------------------------------- 
    
    for it = 1:noIt
        W = sparseapprox(X, Dict, solve, varargin);
        Dict = (X*W')/(W*W');
        Dict = Dict ./ repmat(sqrt(sum(Dict.^2)),[size(Dict,1) 1]);
    end
    
    return;