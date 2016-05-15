%% Input
% Matrix

%% Output
% (R) correlation matrix 
% (E) eigenvalues diagonal matrix
% (V) eigenvectors 
% (M) largest eigenvalue
function [R,E,V,M] = eigencorrelation(X)
    numLines = size(X,1);
    numColumns = size(X,2);
    Y = zeros(numLines,numColumns);
    
    for i = 1:numLines
        stdv = std(X(i,:),1);
        if stdv > 0          
            Y(i,:) = (X(i,:) - mean(X(i,:))) / stdv;	% standardization or unit variation        
        end
    end
    
    R = 1/numColumns*Y*Y';								% estimation of correlation matrix (R)
    [V,E] = eig(R);										% eigenvectors (V) and eigenvalues (E) of correlation matrix
    M(1) = max(diag(E));                                % largest eigenvalue (M)
    
    Ed  = diag(E);    
    for i = 1:size(Ed,1);
        if M(1) == Ed(i)
            M(2) = i;                                   % get the position of largest eigenvalue
        end        
    end