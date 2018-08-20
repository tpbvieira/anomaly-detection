% Function to find the number of users using Akaike equation:
%
% Find m, between 1 to M-1, which minimizes the following quantity:
%
% AIC(m) = -N(M-m) * log( g(m) / a(m) ) + m * (2M - m)
%
% g(m) is the geometric mean of the (M-m) smallest
% eigenvalues of the covariance matrix of observation. - "geomean"
% a(m) is the arithmetic mean. - function "mean"

function [usernumber] = edc_short2(eigenvalues,N)
    [temp,max_size] = size(eigenvalues);
	max_size = max(temp,max_size);
	
    % Putting in the ascendent order. It only works in the ascendent form
	eigenvalues = eigenvalues.';
	eigenvalues = sort(eigenvalues,'ascend');
	c_N = sqrt(N*log(log(N)));
	
    for ii = 1:max_size
		kk = max_size - ii + 1;
		trecho = eigenvalues(1:kk);
		gii = geomean(trecho);
		aii = mean(trecho);
		% mm must vary from 0 to M - 1
		% ii starts with 1 and ends in M
		% wich means M-1 until 0
		mm = max_size - kk;
		EDC_values(ii) = -2 * N * ( max_size - mm ) * log(gii/aii) + mm * ( 2 * max_size - mm) * c_N;
    end
    
	[EDC_min,position] = min( EDC_values (1:max_size) );
	usernumber = position - 1;