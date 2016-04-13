%% Input
% Matrix

%% Output
% (S) covariance matrix 
% (E) eigenvalues diagonal matrix
% (V) eigenvectors 
% (M) largest eigenvalue
function [S,E,V,M] = eigencovariance(X)
    numLines = size(X,1);
    numColumns = size(X,2);
    Y = zeros(numLines,numColumns);
    
    for i = 1:numLines;
        Y(i,:) = (X(i,:) - mean(X(i,:)));	% zero mean
    end
    
    S = 1/numColumns*Y*Y';					% estimation of covariance matrices (S)    
    [V,E] = eig(S);							% eigenvectors (V) and eigenvalues diagonal matrix (E)
    M(1) = max(diag(E));                    % largest eigenvalue (M)
    
    Ed  = diag(E);    
    for i = 1:size(Ed,1);
        if M(1) == Ed(i)
            M(2) = i;                                   % get the position of largest eigenvalue
        end        
    end