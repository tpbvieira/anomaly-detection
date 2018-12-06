function [R,E,V,M] = eigencorrelation(X,method)
% eigencorrelation calculates the eigenvalues, eigenvectors, largest
% eigenvalue using the covariance matrix of zero mean and unit variance of
% X
%
% SYNOPSIS: calculates the eigenvalues, eigenvectors, largest eigenvalue 
% using the correlation matrix from the data X
%
% INPUT X: data matrix
%       method: decomposition method, such as 'eig', 'svd', 'rsvd',
%       'rSVDbasic',  'rSVDsp' e 'rSVD_exSP'
%
% OUTPUT R: correlation matrix 
%        E: eigenvalues diagonal matrix
%        V: right eigenvector matrix
%        M: largest eigenvalue and its index
%
% EXAMPLE: 
%   [R,E,V,M] = eigencorrelation(X,'eig')
%   [R,E,V,M] = eigencorrelation(X,'rsvd')
%   [R,E,V,M] = eigencorrelation(X) 
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

for i = 1:numLines
    stdv = std(X(i,:),1);
    if stdv > 0
        % standardization, have zero mean and unit variance
        Y(i,:) = (X(i,:) - mean(X(i,:))) / stdv;
    end
end

% estimation of correlation matrix (R), with conjugate transpose
R = 1/numColumns*Y*Y';                                                     

% right eigenvectors (V) and eigenvalues (E) of correlation matrix
switch method		
    case 'eig'
        [V,E] = eig(R);                                                     
    case 'svd'
        [U,E,V] = svd(R,'econ');
    case 'rsvd'
        [U,E,V] = rsvd(R,1);
    case 'rSVDbasic'
        [U,E,V] = rSVDbasic(R,1);
    case 'rSVDsp'
        [U,E,V] = rSVDsp(R,1);
    case 'rSVD_exSP'
        [U,E,V] = rSVD_exSP(R,1);
    otherwise
        [V,E] = eig(R);
end

% M(1) value of the largest eigenvalue (M)
M(1) = max(diag(E));
Ed  = diag(E);    
for i = 1:size(Ed,1)
    if M(1) == Ed(i)
        % M(2) index of the largest eigenvalue
        M(2) = i;
    end        
end