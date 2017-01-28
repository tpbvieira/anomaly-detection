function  p = dictprop(D, gap, rad)
% dictprop        Return properties for a normalized dictionary (frame)
% A struct with some selected properties is returned.
% betaij is angle between two dictionary columns, d_i and d_j
%
% p = dictprop(D, gap, rad);
%----------------------------------------------------------------------
% p     A struct with the different properties
% D     The dictionary as a NxK matrix, each element (atom) is a column
%       the dictionary should be normalized.
% gap   true (default) if betagap, decay, and xgap is included, i.e. the
%       properties which are hard to calculate and thus estimated here
% rad   false (default) to return beta-properties as angles in degrees
%       true to return beta-properties as angles in radians
% -- simple information for the D matrix (dictionary)
% p.N        size(D,1)
% p.K        size(D,2)
% p.class    class, supposed to be 'double'
% p.nNaN, p.nInf, p.nnz   number of NaNs, Infs and non-zeros in D
%            if any element of D is NaN or Inf, no properties will be found
% -- the properties easy to calculate are:
% p.lambda   the N eigenvalues of the frame operator (S = D*D'), these are
%            also the non-zero eigenvalues of the Gram matrix (G = D'*D)
% p.A        lower frame bound, smallest non-zero of lambda
% p.B        upper frame bound, largest value of lambda
% p.betamin  smallest betaij
% p.betaavg  average angle for all atoms to closest neighbor
% p.betamse  angle in degrees, cos(betamse) = sqrt(sum(lambda.^2))/K
% p.mu       mutual-coherence, = cos(betamin)
% p.mumin    = p.mu, mutual-coherence, = cos(betamin)
% p.muavg    average coherence, average of cos-values for each atom to its
%            closest neighbor. Note:  muavg <= cos(betaavg)
% p.mumse    = cos(betamse), if A==B then mumse = 1/sqrt(N)
% p.babel    babel function (see Elad), mu1(p), found for p=1,2,...,K-1
% -- the properties hard to calculate           
% p.betagap  let betax be the angle between a vector x and the closest 
%            dictionary atom, betagap is the largest betax for all x (in 
%            the columnspace of D). This property is estimated here,
%            the actual betagap may be larger (but not smaller).
% p.mugap    = cos(betagap) , this is the property denoted as
%            'dictionary distribution ratio' (Quiang 04, Mallat 93)
% p.rgap     = sin(betagap) (r1max in the obsolete FrameProperies.m)
% p.decay    is the decay factor (used by Elad),  decay = mugap^2 
% p.xgap     the most 'distant' vector found. 'better' ones may exist!
% -- other properties not found here
% p.spark    smallest number of columns that are linearly dependent
%----------------------------------------------------------------------
% N = 20; K = 50; D = randn(N,K); D = D-repmat(mean(D), N, 1);
% D = D ./ repmat(sqrt(sum(D.^2)), N, 1);
% p = dictprop(D);

%----------------------------------------------------------------------
% Copyright (c) 2011.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
%
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  29.01.2011  Made function
%  ...
% Ver. 1.5  12.08.2011  removed muab1, muab2, and ratio12 
%                       added rgap and corrected decay to match Elad's def.
%                       removed the eig properties (= mse properties)
%----------------------------------------------------------------------

% A more advanced function is: ..\FrameTools\FrameProperties.m
% this is an older function that did almost the same:
% [fbA, fbB, betamin, betaavg, betamse] = FrameProperties(D, [1,2,7,8,9]);
           
if (nargin < 2); gap = true; end;
if (nargin < 3); rad = false; end;
[N,K] = size(D);
p = struct( 'N',N, 'K',K, 'class',class(D), 'nNaN',nnz(isnan(D)), ...
    'nInf',nnz(isinf(D)), 'nnz',nnz(D) );
if rad
    p.angle = 'radians';
    angleFactor = 1;
else
    p.angle = 'degrees';
    angleFactor = 180/pi;
end

if (p.nNaN == 0) && (p.nInf == 0)
    % S = D*D';   % frame operator
    [U,Lambda] = eig(D*D');              % (D*D')*U = U*Lambda    
    [p.lambda, I] = sort(diag(Lambda),'descend');
    U = U(:,I);                          % (D*D')*U = U*diag(lambda)
    p.B = p.lambda(1);
    iA = numel(p.lambda);
    while (p.lambda(iA) < 1e-12); iA = iA-1;  end;
    p.A = p.lambda(iA);
    p.mumse = sqrt(sum(p.lambda.^2))/K;   % 1/sqrt(N) <= mumse <= 1
    p.betamse = acos(p.mumse)*angleFactor;
    %
    % G = D'*D;   % Gram matrix, G(i,j) = cos( beta_ij )
    Ga = abs(D'*D-eye(K));  
    p.mu = max(Ga(:));
    p.mumin = p.mu;
    p.betamin = acos(p.mu)*angleFactor;
    p.betaavg = mean( acos(max(Ga))*angleFactor );
    p.muavg = mean( max(Ga) );
    Ga = sort(Ga,'descend');   % now sorted
    p.babel = zeros(K-1,1);
    for k=1:(K-1)
        p.babel(k) = max(Ga(1,:));
        Ga(1,:) = Ga(1,:) + Ga(k+1,:);
    end
    % 
    if gap
        % betegap (and decay) must be estimated
        if N <= 4
            keep = 20;
            L = 2000;
            X = randn(N,L);
        else  % build X form eigenspace
            L = min(10*N, 2000);    % number of vectors in X, X is NxL
            if (p.B / p.A) > 10
                keep = min(15, max(floor(N/2),4));  % number of vectors to keep in X
                Lini = min(5, max(2,floor(N/4))); % start with up to Lini vectors from U
            else
                keep = min(30, max(floor(N/2),4));  % number of vectors to keep in X
                Lini = min(12, max(2,floor(N/4))); % start with up to Lini vectors from U
            end
            if iA > Lini
                X = U(:,(iA-Lini+1):iA)*randn(Lini,L);
            else
                X = U(:,1:iA)*randn(iA,L);  % or all
            end
        end
        X = X ./ repmat(sqrt(sum(X.^2)), N, 1);   % normalize
        p.mugap = 1;  count = 0;
        for temp=1:20000;   % while 1   (but not infinitly many times)
            Ga = abs(D'*X);
            [mga,I] = sort(max(Ga), 'ascend');  % smallest first
            % disp(mga(1));
            if (mga(1) >= p.mugap) 
                count = count+1;
                if (count > 10); break; end;
            else
                p.mugap = mga(1);
                p.xgap = X(:,1); %  is most 'distant' vector
            end
            X(:,(keep+1):L) = X(:,I(1:keep))*randn(keep,L-keep);   % linear combination of these
            X = X ./ repmat(sqrt(sum(X.^2)), N, 1);   % normalize
        end
        p.betagap = acos(p.mugap)*angleFactor;
        p.decay = p.mugap^2;
        p.rgap = sqrt(1-p.decay);
    end
else  % nan or inf
    p.lambda = nan(N,1);
    p.A = nan; 
    p.B = nan;
    p.betaeig = nan; 
    p.betamin = nan; 
    p.betaavg = nan;
    p.mu = nan;
    p.mumin = p.mu;
    p.muavg = nan;
    p.mueig = nan;
    p.babel = nan(K-1,1);
    if gap
        p.mugap = nan;
        p.rgap = nan;
        p.decay = nan;
        p.betagap = nan;
        p.xgap = nan(N,1);
    end
end


return;

% -- properties for how close to each other the central values of lambda are 
%    this try to decorrelate mueig from A and B, 
%    fix A and B: what is max and min values for the mueig property 
% p.muab1    mueig for maximal spread of the other lambda values
% p.muab2    mueig for minimal spread of lambda values, when all except A 
%            and B are equal to each other. This is for spread dictionary atoms
% p.ratio12  relative position of mueig between muab1 and muab2, n
%            not relevant for a tight dictionary, since muab1=muab2
%     % March 30, 2011: add muab1, muab2
%     % Aug 12, 2011: removed again as something is not quite correct
%     nB = floor(K/(p.B+p.A));
%     nA = min( floor( (K-nB*p.B)/p.A )-1, iA-nB-1);
%     % max value for mueig given A (and iA) and B 
%     % case 1 is when the eigenvalues are maximal spread given A and B
%     p.muab1 = sqrt( nB*p.B^2 + nA*p.A^2 + (K-nB*p.B-nA*p.A)^2 )/K;  
%     % p.betaab1 = acos(p.muab1)*angleFactor;  % min angle
%     % min value for mueig given A (and iA) and B 
%     % case 2 is when the eigenvalues are mimimum spread given A and B
%     % (the remaining dimensions form a tight frame)
%     p.muab2 = sqrt( p.B^2 + p.A^2 + (iA-2)*( (K-p.B-p.A)/(iA-2) )^2 )/K;  
%     % p.betaab2 = acos(p.muab2)*angleFactor;  % max angle
%     if (abs(p.muab1-p.muab2) < 1e-10); % (p.B > (p.A+1e-6))
%         % give how relative far from ab1 these values are, i.e how
%         % close to a tight frame the remaining dimensions are
%         % 0 for close to case 1 then increasing to 1 for close to case 2
%         p.ratio12 = (p.muab1-p.mueig)/(p.muab1-p.muab2);   % in range 0-1
%         % p.beta12 = (p.betaeig-p.betaab1)/(p.betaab2-p.betaab1);   % in range 0-1
%     else
%         p.ratio12 = nan;
%     end
%     %

