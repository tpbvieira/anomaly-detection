function V = nonzerovec(X)
	j = 0;
	for i = 1:size(X)(1)
		if (X(i)) >= 0
			j = j + 1;
			V(j) = X(i);
		endif
	endfor
