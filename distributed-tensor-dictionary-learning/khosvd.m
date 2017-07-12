function [A_hat,S_hat] = khosvd(noIt, X, A_hat, M1, M2, solver, varargin)
    % What I suggest is to start from the K-SVD algorithm you have and only 
    % replace the matrix rank-one approximation (SVD-based) by the tensor-based 
    % rank-one approximation (HOSVD-based). If the data is multidimensional, 
    % the HOSVD-based rank-one approximation cannot be worse than the SVD-based 
    % one. It is either equal or better. This way you make sure that the 
    % algorithms are actually comparable.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% KHOSVD                                                              %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Author: Florian Roemer, DVT, TU Ilmenau, Nov 2014                   %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % General goal: given Y find A and S such that Y \approx A*S, S is N x T
    %   and sparse, A is M x N and has unit-norm columns.
    %
    % Here, A has a separable structure, i.e., it can be written as A1 \kron A2
    % This structure is exploited using tensors.
    %
    % For details, please see:
    % F. R�mer, G. Del Galdo, and M. Haardt, Tensor-based algorithms for
    % learning multidimensional separable dictionaries, in Proc. IEEE Int.
    % Conf. Acoustics, Speech and Sig. Proc. (ICASSP 2014), Florence, Italy,
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input: 
    %    MaxIT  = maximum number of iterations
    %    Y      = M x T training data.
    %    A_hat  = initial guess for the M x N dictionary A
    %    M1     = size of A1
    %    M2     = size of A2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    K = size(A_hat,2);    
    for it = 1:noIt                                                         % outer loop over iterations        
        S_hat = sparseapprox(X, A_hat, solver, varargin);                   % Step 1: solve sparse recovery problem, i.e., find sparse coeficient matrix S_hat such that Y \approx A_hat*S_hat. Use your favorite sparse solver (OMP/MP. BPDN/Lasso etc
        R = X - A_hat * S_hat;                                              % Step 2: update of dictionary                                                                            
        for k=1:K
            I = find(S_hat(k,:));                                           % for every atom, we find the support, i.e., the training samples where it participates (nonzero elements)
            if ~isempty(I)
                Ri = R(:,I) + A_hat(:,k) * S_hat(k,I);
                % In the K-SVD what follows here is a rank-one MATRIX approximation via the truncated SVD. We use a rank-one TENSOR approximation instead:
                Kn = size(I, 2);
                Yn_tensor = permute(reshape(Ri.',[Kn,M2,M1]),[3,2,1]);      % make it a tensor (inverse 3-mode unfolding of Y^T). compute dominant singular vectors:
                [U1,~] = svd(reshape(Yn_tensor,[M1,M2*Kn]));                % 1-mode unfolding
                u11 = U1(:,1);                  
                [U2,~] = svd(reshape(permute(Yn_tensor,[2,1,3]),[M2,M1*Kn])); % 2-mode unfolding
                u21 = U2(:,1);                  
                [U3,~] = svd(Ri.');                                         % 3-mode unfolding
                u31 = U3(:,1);
                s111 = u31' * Ri.' * conj(kron(u11,u21));                   % compute top-left-front core value
                                
                % rank-one approximation complete. The factors are u11, u21, s111*u31
                A_hat(:,k) = kron(u11,u21);                                  % store results into A and S
                S_hat(k,I) = s111 * u31.';
                R(:,I) = Ri - A_hat(:,k) * S_hat(k,I);
            end
        end
    end