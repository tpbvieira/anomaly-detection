function usernumber = eft_short(eig_values,eft_coeff,M,N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EFT algorithm for Model Order Selection using coefficients eta              %
% Joao Paulo C. L. da Costa                                                   %
% Prof Martin Haardt                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numb_eig = min([M N]);
for pp = 1:(numb_eig-1)
    pp_aux = pp + 1;
    a_mi_ff = 225/((pp_aux^2+2)^2);
    a_mi_fs = 180*pp_aux / ( N * (pp_aux^2 - 1) * (pp_aux^2 +2) );
    a_mi = (a_mi_ff - a_mi_fs)^0.5;
    a_ex_fr = 15/(pp_aux^2 + 2);
    a_diff = a_ex_fr - a_mi;
    a_final(pp_aux) = (0.5*a_diff)^0.5;
    r_final(pp_aux) = exp(-2*a_final(pp_aux));
    J_final(pp_aux) = ( 1 - r_final(pp_aux) ) / ( 1 - (r_final(pp_aux)^pp_aux) );
end
flag_noise = 1;
usernumber = 0;
eig_values = sort(eig_values,'descend');
for pp = 1:(numb_eig-1)
    if (flag_noise == 1)
        pp_aux = pp + 1;
        sum_eig = sum(eig_values((numb_eig-pp):numb_eig));                
        comp_q = eig_values(numb_eig-pp)/sum_eig;
        comp_q_tot = comp_q/J_final(pp_aux) - 1;
        if  (eft_coeff(pp) >= comp_q_tot )
        else
            usernumber = numb_eig - pp;
            flag_noise = 0;            
        end
    end
end