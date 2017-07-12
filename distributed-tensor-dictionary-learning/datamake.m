function X = datamake(D, L, S, snr, met, wlo, whi, lim)
    % datamake        Generate a random data set using a given dictionary
    %
    % X = datamake(D, L, S, snr, 'U', wlo, whi);     
    % X = datamake(D, L, S, snr, 'G', wme, wsi, lim);     
    %-----------------------------------------------------------------------------------
    % arguments:
    %   X      the data matrix, X = D*W  (where W is KxL and sparse)
    %   D      the dictionary, a NxK matrix
    %   L      the size of X is NxL
    %   S      number of non-zeros in each column of W, the positions of the 
    %          non-zeros are random and independent distributed.
    %   snr    SNR of added white Gaussian noise, default is inf (no noise)
    %   met    How the values of the non-zero elements in W are distributed 
    %          'U' - iid uniform distributed entries in range wlo (-1) to whi (1)
    %          'G' - iid Gaussian distributed entries with zeros mean
    %   wlo    Smallest value for weights when uniform distributed, default -1
    %   whi    Largest value for weights when uniform distributed, default 1
    %   wme    Mean value for weights when Gaussian distributed, default 0
    %   wsi    Std for weights when Gaussian distributed, default 1
    %   lim    Limit for small values  when Gaussian distributed, default 0
    %-----------------------------------------------------------------------------------

    if (nargin < 2)
       error('datamake: wrong number of arguments, see help.');
    end
    if (nargin < 3); S = 3; end;
    if (nargin < 4); snr = inf; end;
    if (nargin < 5); met = 'U'; end;
    if (nargin < 6); wlo = -1; end;
    if (nargin < 7); whi = 1; end;
    if (nargin < 8); lim = 0; end;
    [N,K] = size(D);

    X=zeros(N,L);

    % now find the values of the non-zero weights
    if strcmpi(met(1),'U');
        W = (whi-wlo)*rand(S,L) + wlo;
    elseif strcmpi(met(1),'G');
        if (nargin < 6); wlo = 0; end;
        if (nargin < 7); whi = 1; end;
        W = wlo + whi*randn(S,L);
        if (lim > 0)
            I = find(abs(W)<lim);
            while (numel(I)>0)
                W(I) = wlo + whi*randn(numel(I),1);
                I = find(abs(W)<lim);
            end
        end
    end

    % make X
    for i=1:L
        [temp,J] = sort(rand(K,1));
        X(:,i) = D(:,J(1:S)) * W(:,i);
    end

    % add noise
    if (snr < 200)
        stdSignal = std(X(:));
        stdNoise  = stdSignal * power(10,-snr/20);
        X = X + stdNoise * randn(N,L);                                      % add noise to signal, always Gaussian
    end

    return