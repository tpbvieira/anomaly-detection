% Function to find the number of users using Akaike equation:
% Find m, between 1 to M-1, which minimizes the following quantity:
%
% AIC(m) = -N(M-m) * log( g(m) / a(m) ) + m * (2M - m)
% g(m) is the geometric mean of the (M-m) smallest eigenvalues of the covariance matrix of observation. - "geomean"
% a(m) is the arithmetic mean. - function "mean"
function [usernumber] = akaike_short2(eigenvalues_vector,N);
    [temp,M] = size(eigenvalues_vector);
	M = max(temp,M);
	% Putting in the ascendent order
	% It only works in the ascendent form
	eigenvalues_vector = eigenvalues_vector.';
	eigenvalues_vector = sort(eigenvalues_vector,'ascend');
	for ii = 1:M
		sample_size = M - ii + 1;
		sample = eigenvalues_vector(1:sample_size);
		gii = geomean(sample);
		aii = mean(sample);
		% mm must vary from 0 to M - 1
		% ii starts with 1 and ends in M
		% wich means M-1 until 0
		mm = M - sample_size;
		AIC_values(ii) = -2*N*(M - mm)*log(gii/aii) + 2*mm*(2*M - mm);
	end
	[AIC_min,AIC_min_index] = min(AIC_values(1:M));
	usernumber = AIC_min_index - 1;
