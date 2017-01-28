function  [d, I2] = dictdiff(D1, D2, met, vdh)
% dictdiff        Return a measure for the distance between two dictionaries
%                 Several measures are possible.
% An alternative is the norm of the difference, but then both the order and
% the sign of the columns should be taken into account, let I2 be the
% indexes for the colums in dictionary 2 which gives the optimal assignment
% relative to dictionary 1, then a norm differeence can be:
%   diffD1D2 = norm(D1-D2(:,I2), 'fro');
%
% Examples using this function
% d = dictdiff(D1, D2);  % beta(D1,D1) measure, mean angle to closest
%                        % vector in the other dictionary
% d = dictdiff(D1, D2, 'mean-1');    % avstand er 2-norm
% [d, I2] = dictdiff(D1, D2, 'mean', 'beta'); 
% d = dictdiff(D1, D2, 'mean', @(v1, v2) sum(abs(v1(:) - v2(:))));
% d = dictdiff(D1, D2, 'mean', @(v1, v2) norm(v1-v2,inf) )
%
% [d, I2] = dictdiff(D1, D2, met, vdh);   
% -------------- arguments ----------
%  d   : the returned measure for difference, if angle in radians
%  I2  : order for columns in D2 for the optimal assignment, D2(:,I2)
%  D1  : First dictionary, size NxK1
%  D2  : Second dictionary, size NxK2
%  met : the method to use, default 'mean'
%        'all'    - returns a vector d of size (K1+K2)x1, each element is
%                   distance to closest vector in the other dictionary.
%        'all-1'  - When K=K1=K2, returns a vector d of size Kx1 
%                   d(k) = vdh( D1(:,k), D2(:,I2(k)) )
%        'mean'   - mean of d as for 'all'
%        'mean-1' - mean of d as for 'all-1'
%        'max'    - max of d as for 'all'
%        'max-1'  - max of d as for 'all-1'
%  vdh : the vector dictance function, a function handle or a string
%        which defines how distance between two vectors should be
%        calculated, default is inner angle between them (beta)
%        ex: vdh = @(v1, v2) norm(v1-v2,1);  % 1-norm
%        Some pre-defined alternatives are:
%        'beta'   inner angle between two vectors, 0 <= beta <= pi/2
%        'theta'  angle between two vectors, 0 <= theta <= pi
%        'sin'    sine of angle between two vectors
%        'norm2'  Euclid distance:  vdh = @(v1, v2) norm(v1-v2,2);
%----------------------------------------------------------------------
% N=4;K=8; D1 = randn(N,K); D2 = randn(N,K);
% N=4;K=8; D1 = randn(N,K); D2 = D1(:,randperm(K)) + 0.1*randn(N,K);
% d = dictdiff(D1, D2, 'all');    
% [d2, I2] = dictdiff(D1, D2, 'all-1');    

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger, Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  29.04.2009  KS: test started
% Ver. 1.1  04.04.2011  KS: some minor updates
%----------------------------------------------------------------------

mfile = 'dictdiff';
d = inf; I2 = 1; 

if (nargin < 2)
    disp([mfile,' wrong number of input arguments, see help.']);
    return;
end
if (nargin < 3)
    met = 'mean';
end
if ~( strcmp(met,'all')  || strcmp(met,'all-1')  || ...
      strcmp(met,'mean') || strcmp(met,'mean-1') || ...
      strcmp(met,'max')  || strcmp(met,'max-1')  )
    disp([mfile,': met (',met,') is not valid, see help.']);
    return;
end
if (nargin < 4)
    vdh = 'beta';
    % vdh = @(v1, v2) norm(v1-v2);
end
if ischar(vdh)
    if (strcmp(vdh, 'theta'))
        vdh = @(v1, v2) acos((v1(:)'*v2(:)) / (sqrt(v1(:)'*v1(:))*sqrt(v2(:)'*v2(:))));
    elseif (strcmp(vdh, 'thabs')) || (strcmp(vdh, 'beta'))
        vdh = @(v1, v2) acos(abs((v1(:)'*v2(:)) / (sqrt(v1(:)'*v1(:))*sqrt(v2(:)'*v2(:)))));
    elseif (strcmp(vdh, 'sin'))
        vdh = @(v1, v2) sqrt(1 - ((v1(:)'*v2(:))^2)/(((v1(:)'*v1(:))*(v2(:)'*v2(:)))));
    elseif (strcmp(vdh, 'norm2')) || (strcmp(vdh, 'norm')) || (strcmp(vdh, '2norm'))
        vdh = @(v1, v2) norm(v1-v2);
    end
end
if (~strcmp(class(vdh), 'function_handle'))
    disp([mfile,' Use inner angle measure for distance between vectors.']);
    vdh = @(v1, v2) acos(abs((v1(:)'*v2(:)) / (sqrt(v1(:)'*v1(:))*sqrt(v2(:)'*v2(:)))));
end

% control arguments
[N, K1] = size(D1);
[N2, K2] = size(D2);
if (N ~= N2)
    disp([mfile,' length of columns in D1 and D2 do not match.']);
    return;
end

% matrix of distances
dist = zeros(K1,K2);
for k1=1:K1
    for k2=1:K2
        dist(k1,k2) = vdh(D1(:,k1),D2(:,k2));
    end
end
[Y1,I1] = min(dist,[],1);
Y1 = Y1'; I1 = I1';          % Y1 = diag(d(I1,:))     
[Y2,I2] = min(dist,[],2);    % Y2 = diag(d(:,I2))     
d = [Y1; Y2];

% find order of column vectors in D2
if ( (K1==K2) && (strcmp(met((end-1):end),'-1')) )
    if (sum(abs((I1(I2)-(1:K1)'))) == 0)  % ?  ?
        % disp([mfile,': ok.']);
    else
        I2 = assignmentoptimal(dist);   
    end
    d = diag(dist(:,I2));
end

if ( strcmp(met,'mean') || strcmp(met,'mean-1'))
    d = mean(d);
elseif ( strcmp(met,'max') || strcmp(met,'max-1'))
    d = max(d);
end

return;

