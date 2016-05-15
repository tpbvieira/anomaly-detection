function d_est = ranoi_app(eig_values);
% Discriminant approach
eig_values = sort(eig_values,'descend');
eig_num = length(eig_values);
% Calculating mu_fct
%
%for ii = 1:M
%    kk = M - ii + 1;
%    trecho = eig_vec(1:kk);
%    gii = geomean(trecho);
%    aii = mean(trecho);
%end
%
for kk = 1:(eig_num-1)
    ii = eig_num - kk + 1;;
    trecho = eig_values(1:ii);
    gii = geomean(trecho);
    aii = mean(trecho);
    %mu_fct(kk) = (1/(eig_num-kk))*sum(eig_values(kk+1:eig_num));
    mu_fct(kk) = geomean(trecho)/mean(trecho);
end
%
% Calculating alpha_c
for kk = 1:(eig_num-1)
    trecho = eig_values(1:kk);  
    alpha_var(kk) = (geomean(trecho) - mu_fct(kk))/mu_fct(kk);
end
% alpha_const is the position of the element
[temp,alpha_const] = max(alpha_var);
%
% Calculating ksi
for kk = 1:(eig_num-1)
    trecho = eig_values(1:kk);  
    ksi(kk) = 1-alpha_const*(geomean(trecho)-mu_fct(kk))/mu_fct(kk);
end
%
for kk = 1:(eig_num-1)
    trecho = eig_values(1:kk);  
    discr_1(kk) = geomean(trecho)/mean(trecho);
    %
    discr_2(kk) = ksi(kk)/sum(ksi(1:(eig_num-1)));
    %
    discr_crit(kk) = discr_1(kk) - discr_2(kk);
    %
end
%
%
[temp2,d_est]= min(discr_crit);