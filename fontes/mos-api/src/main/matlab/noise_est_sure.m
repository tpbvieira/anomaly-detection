function [sigma] = noise_est_sure(eig_values,M,N);
%
% Taking the 25th percentile of l
%
num_eig = min(M,N);
%
% It is considered M < N
%
eig_desc = sort(eig_values,'descend');
%
gamma = N/M;
%
b_mp = (1+gamma^(-0.5))^2;
%
for ii = 1:num_eig
    EDF_prob(ii) = (num_eig-ii+1)/num_eig;
    factor_correc(ii) = mp_density(EDF_prob(ii),gamma);
    eig_corr_eins(ii) = eig_desc(ii)/factor_correc(ii);
end
% Taking the 25th percentile
percent_25 = round(0.25*num_eig);
%
eig_selection = eig_corr_eins((num_eig-percent_25):num_eig);
%
sigma_sqr_eins = median(eig_selection);
%
% Normalizing the eigenvalues:
%
norm_eig_eins = eig_desc/sigma_sqr_eins;
%
% Crude estimation for d
mask_norm = norm_eig_eins > b_mp;
% Give the position of the positive normalized eigenvalue closest to b_mp
r_cru = sum(mask_norm);
%
for ii = (r_cru+1):num_eig
    kk = ii - r_cru;
    EDF_prob_zwei(kk) = (num_eig-ii+1)/(num_eig-r_cru);
    factor_correc_zwei(kk) = mp_density(EDF_prob_zwei(kk),gamma);
    eig_corr_zwei(kk) = eig_desc(ii)/factor_correc_zwei(kk);
end
%
sigma = median(eig_corr_zwei);
%