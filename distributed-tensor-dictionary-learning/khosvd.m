function A_hat = khosvd(MaxIT, Y, A_hat, M1, M2, solver, varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% KHOSVD example                                                      %%%
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

    T = size(Y,2);
    N = size(A_hat,2);
    M = [M1,M2];

    % outer loop over iterations
    for nIt = 1:MaxIT
        % Step 1: solve sparse recovery problem, i.e., find sparse S_hat such
        % that Y \approx A_hat*S_hat. Use your favorite sparse solver here,
        % e.g., OMP/MP or BPDN/Lasso or anything you like.    
        S_hat = sparseapprox(Y, A_hat, solver, varargin);
        
        % Step 2: update of dictionary
        % we loop through the atoms (in a random order)
        for n = randperm(N)
            % for every atom, we find the support, i.e., 
            % the training samples where it participates (nonzero elements)
            supportCols = find(S_hat(n,:));                                     % to fight rounding errors, consider using a threshold here
            % is it active at all?
            if ~isempty(supportCols)           
                % subtract contributions from all the other atoms
                notn = [1:n-1,n+1:N];
                Yn = Y(:,supportCols) - A_hat(:,notn) * S_hat(notn,supportCols);
                Kn = size(supportCols, 2);
                % this matrix should be rank one. In the K-SVD what follows
                % here is a rank-one matrix approximation via the truncated
                % SVD. We use a rank-one tensor approximation instead:

                % make it a tensor (inverse 3-mode unfolding of Y^T)
                Yn_tensor = permute(reshape(Yn.',[Kn,M2,M1]),[3,2,1]);
                % compute dominant singular vectors
                  % 1-mode unfolding
                [U1,~] = svd(reshape(Yn_tensor,[M1,M2*Kn]));                 
                u11 = U1(:,1);
                  % 2-mode unfolding
                [U2,~] = svd(reshape(permute(Yn_tensor,[2,1,3]),[M2,M1*Kn]));
                u21 = U2(:,1);
                  % 3-mode unfolding 
                [U3,~] = svd(Yn.');u31 = U3(:,1);            
                % compute top-left-front core value
                s111 = u31'*Yn.'*conj(kron(u11,u21));
                % rank-one approximation complete. The factors are u11, u21, s111*u31

                % store results into A and S
                A_hat(:,n) = kron(u11,u21);
                S_hat(n,supportCols) = s111*u31.';
            end
        end     
    end