function D = makedict(N, K, met)
    % Make a random dictionary, iid Gaussian or uniform entries
    % met 'U' (default) should be like Kreutz-Delgado, Engan, et al. in 2003 
    % and Aharon et al. in 2005.
    % The dictionary is normalized, each column scaled so that its 2-norm == 1
    %
    % D = dictmake(N, K);           % met 'U' is default
    % D = dictmake(N, K, met);     
    %-----------------------------------------------------------------------------------
    % arguments:
    %   D      the resulting dictionary, a NxK matrix
    %   N, K   the size of D is NxK
    %   met    How the random elements in D are distributed (before normalization)
    %          'U' - iid uniform distributed entries in range -1 to 1
    %          'u' - iid uniform distributed entries in range  0 to 1
    %          'G' - iid Gaussian distributed entries with zeros mean
    %-----------------------------------------------------------------------------------

    if (nargin < 2)
       error('dictmake: wrong number of arguments, see help.');
    end
    
    if (nargin < 3)
        met = 'U';
    end;

    % make the generating frame/dictionary
    if met(1)=='U';
        D=2*rand(N,K)-1;
    elseif met(1)=='u';
        D=rand(N,K);
    elseif met(1)=='G';
        D=randn(N,K);
    end
    
    % normalize
    D = D.*(ones(size(D,1),1)*(1./sqrt(sum(D.*D))));

    return