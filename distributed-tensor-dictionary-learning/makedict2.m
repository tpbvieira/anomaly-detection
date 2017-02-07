function [A, A1, A2] = makedict2(M1, M2, N1, N2, met)
    % Make a random dictionary from kroneker product, iid Gaussian or uniform entries
    % met 'U' (default) should be like Kreutz-Delgado, Engan, et al. in 2003 
    % and Aharon et al. in 2005.
    % The dictionary is normalized, each column scaled so that its 2-norm == 1
    %
    % A = dictmake(M1, M2, N1, N2);           % met 'U' is default
    % A = dictmake(M1, M2, N1, N2, met);     
    %-----------------------------------------------------------------------------------
    % arguments:
    %   A               the resulting dictionary, a NxK matrix
    %   M1,M2,N1,N2     the size of D is M1xN1 x M2xN2
    %   met             How the random elements in D are distributed (before normalization)
    %                   'U' - iid uniform distributed entries in range -1 to 1
    %                   'u' - iid uniform distributed entries in range  0 to 1
    %                   'G' - iid Gaussian distributed entries with zeros mean
    %-----------------------------------------------------------------------------------

    if (nargin < 4)
       error('dictmake: wrong number of arguments, see help.');
    end
    
    if (nargin < 5)
        met = 'U';
    end;

    % make the generating frame/dictionary
    if met(1)=='U';
        A1 = 2 * rand(M1,N1) -1;
        A2 = 2 * rand(M2,N2) -1;
    elseif met(1)=='u';
        A1 = rand(M1,N1);
        A2 = rand(M2,N2);
    elseif met(1)=='G';
        A1 = randn(M1,N1);
        A2 = randn(M2,N2);
    end
    
    % normalize
    %A1 = A1.*(ones(size(A1,1),1)*(1./sqrt(sum(A1.*A1))));
    %A2 = A2.*(ones(size(A2,1),1)*(1./sqrt(sum(A2.*A2))));
    
    A = kron(A1,A2);
    
    return