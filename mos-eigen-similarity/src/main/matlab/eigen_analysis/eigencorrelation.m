function [R,E,V,M] = eigencorrelation(X)
% eigencorrelation calculates the eigenvalues, eigenvectors, largest
% eigenvalue using the covariance matrix of zero mean and unit variance of
% X
%
% SYNOPSIS: calculates the eigenvalues, eigenvectors, largest eigenvalue 
% using the correlation matrix from the data X
%
% INPUT X: data matrix
%
% OUTPUT R: correlation matrix 
%        E: eigenvalues diagonal matrix
%        V: eigenvector matrix
%        M: largest eigenvalue
%
% EXAMPLE eigencorrelation(X)
%
% SEE ALSO 
%
% created with MATLAB R2016a on Ubuntu 16.04
% created by: Thiago Vieira
% DATE: 
%

numLines = size(X,1);
numColumns = size(X,2);
Y = zeros(numLines,numColumns);

for i = 1:numLines
    stdv = std(X(i,:),1);
    if stdv > 0          
        Y(i,:) = (X(i,:) - mean(X(i,:))) / stdv;                            % standardization, have zero mean and unit variance
    end
end

R = 1/numColumns*Y*Y';                                                      % estimation of correlation matrix (R), with conjugate transpose
[V,E] = eig(R);                                                             % eigenvectors (V) and eigenvalues (E) of correlation matrix
M(1) = max(diag(E));                                                        % value of the largest eigenvalue (M)

Ed  = diag(E);    
for i = 1:size(Ed,1)
    if M(1) == Ed(i)
        M(2) = i;                                                           % index of the largest eigenvalue
    end        
end