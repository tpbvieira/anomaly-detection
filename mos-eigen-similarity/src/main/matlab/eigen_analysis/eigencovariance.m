function [S,E,V,M] = eigencovariance(X,method)
% eigencovariance calculates the eigenvalues, eigenvectors, largest
% eigenvalue using the covariance matrix of zero mean of X
%
% SYNOPSIS: eigencovariance calculates the eigenvalues, eigenvectors, 
% largest eigenvalue using the covariance matrix of zero mean of X
%
% INPUT X: data matrix
%       method: decomposition method, such as 'eig', 'svd', 'rsvd',
%       'rSVDbasic',  'rSVDsp' e 'rSVD_exSP'
%
% OUTPUT S: covariance matrix 
%        E: eigenvalues diagonal matrix
%        V: eigenvector matrix
%        M: largest eigenvalue
%
% EXAMPLE: 
%   [S,E,V,M] = eigencovariance(X,'eig')
%   [S,E,V,M] = eigencovariance(X,'rsvd')
%   [S,E,V,M] = eigencovariance(X) 
%
% SEE ALSO 
%
% created with MATLAB R2016a on Ubuntu 16.04
% created by: Thiago Vieira
% DATE: 
%

if nargin<2,
    method='eig';
end

numLines = size(X,1);
numColumns = size(X,2);
Y = zeros(numLines,numColumns);

for i = 1:numLines;
    % zero mean
    Y(i,:) = (X(i,:) - mean(X(i,:)));
end

% estimation of covariance matrices (S)    
S = 1/numColumns*Y*Y';                                                      

% right eigenvectors (V) and eigenvalues (E) of correlation matrix
switch method		
    case 'eig'
        [V,E] = eig(S);                                                     
    case 'svd'
        [U,E,V] = svd(S,'econ');
    case 'rsvd'
        [U,E,V] = rsvd(S,1);
    case 'rSVDbasic'
        [U,E,V] = rSVDbasic(S,1);
    case 'rSVDsp'
        [U,E,V] = rSVDsp(S,1);
    case 'rSVD_exSP'
        [U,E,V] = rSVD_exSP(S,1);
    otherwise
        [V,E] = eig(S);
end                                                           

% value of the largest eigenvalue (M)
M(1) = max(diag(E));                                                        
Ed  = diag(E);    
for i = 1:size(Ed,1);
    if M(1) == Ed(i)
        % index of the largest eigenvalue
        M(2) = i;
    end        
end