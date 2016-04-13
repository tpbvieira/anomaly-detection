%covariance matrix (S)
%eigenvectors (V)
%eigenvalues (E)

%máximo autovalor associado à matriz de covariância (M)
%covariance (Y).
function [S,E,V,M] = covariancia(X)
    numLines = size(X,1);
    numColumns = size(X,2);
    for i = 1:numLines;
        Y(i,:) = (X(i,:) - mean(X(i,:)));	%covariance
    end
    S = 1/numColumns*Y*Y';					%covariance matrix (S)    
    [V,E] = eig(S);							%eigenvectors (V) and eigenvector diagonal matrix (E)
    M = max(diag(E));						%greatest eigenvector