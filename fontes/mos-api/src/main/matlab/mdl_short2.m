function [usernumber] = mdl_short2(eig_vec,N);
%
% Function to find the number of users
% using Akaike equation:
%
% Find m, between 1 to M-1, which minimizes the
% following quantity:
%
% AIC(m) = -N(M-m) * log( g(m) / a(m) ) + m * (2M - m)
%
% g(m) is the geometric mean of the (M-m) smallest
% eigenvalues of the covariance matrix of observation. - "geomean"
% a(m) is the arithmetic mean. - function "mean"
[temp,M1] = size(eig_vec);
M1 = max(temp,M1);
M = M1;
% Putting in the ascendent order
% It only works in the ascendent form
eig_vec = eig_vec.';
eig_vec = sort(eig_vec,'ascend');
for ii = 1:M1
    kk = M1 - ii + 1;
    trecho = eig_vec(1:kk);
    gii = geomean(trecho);
    aii = mean(trecho);
    % mm must vary from 0 to M - 1
    % ii starts with 1 and ends in M
    % wich means M-1 until 0
    mm = M1 - kk;
    MDL_values(ii) = -N * (M - mm) * log(gii/aii) + 0.5* mm * (2*M - mm) * log(N);
end
[MDL_min,position] = min(MDL_values(1:M1));
usernumber = position - 1;
