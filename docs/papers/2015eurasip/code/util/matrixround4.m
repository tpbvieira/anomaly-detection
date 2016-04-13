function [X] = matrixround4(X)
rows = size(X,1);
cols = size(X,2);
for i = 1:rows
  for j = 1:cols
	Y = sprintf('%5.4f', X(i,j));
	X(i,j) = str2num(Y) + 0;
  end
end