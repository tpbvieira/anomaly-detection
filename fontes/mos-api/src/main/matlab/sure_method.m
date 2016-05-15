function d_est = sure_method(X_mat,M,N);
% Model order selection
sing_values = svd(X_mat);
%
eig_values = sing_values.^2;
%
[sigma_rmt] = noise_est_sure(eig_values,M,N);
%
max_d = min(M,N);
%
other_dim = max(M,N);
%
for ii = 1:(max_d-1)
    %
    sigma_r_sqr = (1/(max_d-ii))*sum(eig_values((ii+1):max_d));
    %
    C_part_one = 0;
    C_part_three = 0;
    for jj = 1:ii
        %
        for kk = (ii+1):max_d
            C_part_one = (4*sigma_rmt/other_dim)* (eig_values(jj)-sigma_r_sqr)/(eig_values(jj)-eig_values(kk)) + C_part_one;
        end
        %
        C_part_three = -(2*sigma_rmt/other_dim)*(max_d-1)*(1-sigma_r_sqr/eig_values(jj)) + C_part_three;
        %
    end
    %
    C_part_two = (2*sigma_rmt/other_dim)*ii*(ii-1);
    %
    C_total = C_part_one + C_part_two + C_part_three;
    %
    Risk_part_one = (max_d-ii)*sigma_r_sqr;
    %
    Risk_part_two = 0;
    Risk_part_three = 2*sigma_rmt*ii;
    Risk_part_four = 0;
    Risk_part_five = 0;
    %
    for jj = 1:ii
        %
        Risk_part_two = (sigma_r_sqr^2)/eig_values(jj) + Risk_part_two;
        %       
        Risk_part_four = (-2*sigma_rmt*sigma_r_sqr)/eig_values(jj) + Risk_part_three;
        %
        Risk_part_five = (4*sigma_rmt*sigma_r_sqr/other_dim)/eig_values(jj) + Risk_part_four;
        %
    end
    %
    Risk_total(ii) = Risk_part_one + Risk_part_two + Risk_part_three + Risk_part_four + Risk_part_five + C_total;
end
[value,d_est] = min(Risk_total);