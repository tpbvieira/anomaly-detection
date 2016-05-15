#Cálculo dos coeficientes: EFT_CALC_COEF
function [local_coeff,prob_found]=calc_coef_paper(M,N,Pfa,coeff,q);
	max_eig_numb = min([M N]);
	for pp = 1:(max_eig_numb-1)
		pp_aux = pp+1;
		a_mi_ff = 225/((pp_aux^2+2)^2);
		a_mi_fs = 180*pp_aux/(N*(pp_aux^2-1)*(pp_aux^2+2));
		a_mi = (a_mi_ff-a_mi_fs)^0.5;
		a_ex_fr = 15/(pp_aux^2+2);
		a_diff = a_ex_fr-a_mi;
		a_final(pp_aux) = (0.5*a_diff)^0.5;
		r_final(pp_aux) = exp(-2*a_final(pp_aux));
		J_final(pp_aux)=(1-r_final(pp_aux))/(1-r_final(pp_aux)^pp_aux));
	endfor
	barra = waitbar(0,'Please wait... Calculating thresholds for EFT');
	for ii = 1:q
		noise = randn(M,N);
		noise_eig = (svd(noise).^2)/N;
		noise_eig = sort(noise_eig,'descend');
		for pp = 1:(max_eig_numb-1)
			pp_aux = pp+1;
			sum_eig = sum(noise_eig((max_eig_numb-pp):max_eig_numb));
			comp_q = noise_eig(max_eig_numb-pp)/sum_eig;
			comp_q_tot = comp_q/J_final(pp_aux)-1;
			prob_q(pp,:) = coeff<comp_q_tot;
			if (ii == 1)
				prob_fim(pp,:) = 1*prob_q(pp,:);
			else
				prob_fim(pp,:)=((ii-1)*prob_fim(pp,:)+prob_q(pp,:))/ii;
			endif
		endfor
		waitbar(ii/q,barra);
	endfor
	close(barra);
	for pp = 1:(max_eig_numb-1)
		[temp,local_Pfa(pp)] = min(abs(prob_fim(pp,:)-Pfa));
		local_coeff(pp) = coeff(local_Pfa(pp));
		if (local_Pfa(pp) == 1)
			prob_found(pp) = 0;
			Else
			prob_found(pp) = prob_fim(pp,local_Pfa(pp)-1);
		endif
	endfor
