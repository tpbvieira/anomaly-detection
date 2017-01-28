function shat = omp(y,A,isnormalized,varargin)
    % OMP     Orthogonal Matching Pursuit
    %
    % Syntax:
    %    S = OMP(y,A[,isnormalized])
    %    S = OMP(y,A,isnormalized,'Max_K',K)
    %    S = OMP(y,A,isnormalized,'Threshold_Abs',thrsh_abs)
    %    S = OMP(y,A,isnormalized,'Threshold_Diff',thrsh_diff)
    %
    % Input:
    %    y - observations of size M x 1 (single snapshot case) or M x T
    %        (multiple snapshot case)
    %    A - dictionary (overcomplete basis) providing the sparse
    %        representation of y. In a CS setting, A is the sensing matrix,
    %        i.e., the combination of the meashrement matrix and the basis.
    %    isnormalized - {true,false}: are the columns of A normalized such that
    %        they have identical Euclidean norms? If yes (true), the algorithm
    %        is slightly faster. Defaults to no (false).
    %    Max_K, Threshold_Abs and Threshold_Rel allow to specify stopping
    %        conditions for the algorithm. They are optional but it is highly
    %        specified to specify at least one of them. They can also be mixed.
    %        The three criteria are:
    %           'Max_K' stops when K atoms have been found. Defaults to K = size(A,1).
    %           'Threshold_Abs' stops when the norm of the residual ||r_k||
    %              drops below thrsh_abs. Defaults to thrsh_abs = 0.
    %           'Threshold_Diff' stops when the norm of the relative change of
    %              the residual ||r_k - r_{k+1}|| // ||r_{k+1}|| drops below
    %              thrsh_diff. Defaults to thrsh_diff = 0.
    %
    %
    % Output:
    %    S - sparse vector of size N x 1 (single snapshot case) or N x T
    %        (multiple snapshot case). For T>1, group sparsity in form of
    %        row-sparsity is enforced, i.e., K rows of S are non-zero.
    %
    % Example:
    %   M = 32;N = 128;K = 2;T = 1;
    %   A = randn(M,N);
    %   support = randperm(N,K);
    %   S = zeros(N,T);
    %   S(support,:) = rand(K,T);
    %   y = A*S;
    %   S_hat = OMP(y,A,false,'Max_K',K);
    %   disp(norm(S_hat-S,'fro')^2);
    %   
    % Author:
    %   Florian Roemer, DVT, TU Ilmenau, 2013
    %
    % Revision history:
    %   10/2013: initial version
    %   11/2014: multiple snapshot support added
    %   09/2015: error in normalization fixed (missing sqrt)
    %   09/2016: speed up by iterative update of amplitudes and residual
    %   09/2016: added stopping conditions

    T = size(y,2);
    [M,N] = size(A);

    % default values
    Kmax = M;
    thrsh_abs = eps*norm(y,'fro')*100;
    thrsh_diff = eps*norm(y,'fro')*100;

    % parse paremeters
    if (nargin < 3) || isempty(isnormalized)
        isnormalized = false;
    end
    if ~isempty(varargin)
        use_thrsh_abs = false;
        use_thrsh_diff = false;
        if mod(length(varargin),2) ~= 0
            error('Invalid parameters.');
        end
        for iPar = 1:length(varargin)/2
            P = varargin{2*iPar-1};
            V = varargin{2*iPar};
            switch P
                case 'Max_K'
                    Kmax = V;
                case 'Threshold_Abs'
                    thrsh_abs = V;
                    use_thrsh_abs  = true;
                case 'Threshold_Diff'
                    thrsh_diff = V;
                    use_thrsh_diff = true; 
                otherwise
                    error('Unknown parameter.');
            end
        end
    else
        % default behavior when nothing is passed: terminate when the residual
        % is zero (makes sense for synthetic noise-free data), otherwise for
        % K=M.
        use_thrsh_abs = true;
        use_thrsh_diff = false;
    end

    if ~isnormalized
        ncA = sum(abs(A).^2,1)'; 
        % we might be tempted to normalize A via A = bsxfun(@rdivide,A,ncA);
        % However, then the amplitudes s are wrong and we have to reintegrate
        % the norms there...
        %
        % In case you are missing the sqrt in ncA: switched to squared
        % correlations for c since that is what we should do in the multi
        % snapshot case.
    end

    shat = zeros(N,T);
    active_atoms = 1:N;
    support = [];
    r = y; % initialize residual with observations

    for k = 1:Kmax
        if isnormalized
            c = A(:,active_atoms)'*r;        
            c = sum(abs(c).^2 ,2); % correlate residual to atoms
        else
            c = sum(abs(A(:,active_atoms)'*r).^2 ./ ncA(active_atoms,ones(T,1)),2);
        end
        [~,w] = max(c);                       % pick maximally correlated one
        support = [support,active_atoms(w)];  % add to support
        active_atoms(w) = [];                 % remove from the ones we check

        % % fit coefficients:
        % this fast iterative update replaces shat(support,:) = A(:,support)\y;
        if k == 1
            a = A(:,support);
            v2 = a;
            v2n = v2/(v2'*v2);
            v2y = v2n'*y;
            shat(support,:) = v2y;
            B = v2n';        
        else
            a = A(:,support(end)); % new atom
            v1 = B*a;
            v2 = a - A(:,support(1:end-1))*v1;
            v2n = v2/(v2'*v2);
            v2y = v2n'*y;
            shat(support,:) = shat(support,:) + [-v1;1]*v2y;
            B = [B-v1*v2n';v2n'];
        end
        rold = r;
        r = r - v2*v2y; % update residual 

        % Check stopping conditions
        if use_thrsh_abs 
            rF = norm(r,'fro');
            if rF<=thrsh_abs
                break
            end
        end

        if use_thrsh_diff
            if ~use_thrsh_abs, rF = norm(r,'fro'); end        
            % for synthetic noise-free data this will not work: rF gets
            % essentially zero (a few eps) which blows up this fraction
            if norm(rold-r,'fro')/rF<=thrsh_diff
                break
            end
        end
    end