%correlation matrix (R)
%eigenvalues (E) of correlation matrix
%eigenvectors (V) of correlation matrix 
%maximum eigenvalue (M)
%correlation coefficient or "Pearson's correlation coefficient" (Y)
%map with original data (O)
function [R,E,V,M,Y,O] = correlacao(X)
numLines = size(X,1);
numColumns = size(X,2);
j = 0;

for i = 1:numLines
	if std(X(i,:),1) > 0
		j = j + 1;
        O(j) = i;
		Y(j,:) = (X(i,:) - mean(X(i,:))) / std(X(i,:),1);	%correlation coefficient or "Pearson's correlation coefficient"
    end
end

if j == 0
	Y = [,];
    O = [];
end

R = 1/numColumns*Y*Y';										%correlation matrix (R)
[V,E] = eig(R);												%eigenvectors (V) e eigenvalues (E) of correlation matrix
M = max(diag(E));											%maximum eigenvalue (M)