function k = numpca(E)
	E_sorted = sort(E,'descend');
	k = 0;
	for i = 1:size(E_sorted,1)
		if (sum(E_sorted(1:i))/sum(E_sorted)) >= 0.99
			k = i;
			break;
        end
    end
