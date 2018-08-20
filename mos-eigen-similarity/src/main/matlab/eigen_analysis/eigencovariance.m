function [S,E,V,M] = eigencovariance(X)
% eigencovariance calculates the eigenvalues, eigenvectors, largest
% eigenvalue using the covariance matrix of zero mean of X
%
% SYNOPSIS: eigencovariance calculates the eigenvalues, eigenvectors, 
% largest eigenvalue using the covariance matrix of zero mean of X
%
% INPUT X: data matrix
%
% OUTPUT S: covariance matrix 
%        E: eigenvalues diagonal matrix
%        V: eigenvector matrix
%        M: largest eigenvalue
%
% EXAMPLE eigencovariance(X)
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

for i = 1:numLines;
    Y(i,:) = (X(i,:) - mean(X(i,:)));                                       % zero mean
end

S = 1/numColumns*Y*Y';                                                      % estimation of covariance matrices (S)    
[V,E] = eig(S);                                                             % eigenvectors (V) and eigenvalues diagonal matrix (E)
M(1) = max(diag(E));                                                        % value of the largest eigenvalue (M)

Ed  = diag(E);    
for i = 1:size(Ed,1);
    if M(1) == Ed(i)
        M(2) = i;                                                           % index of the largest eigenvalue
    end        
end