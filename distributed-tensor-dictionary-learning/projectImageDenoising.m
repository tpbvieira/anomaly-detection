%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Evaluates Dictionary Learning methods for image denoising       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: Thiago Vieira, UnB, 2017                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% General goal: 
%	MOD
%	RLS-DLA
%	K-SVD
%	K-HOSVD
%	T-MOD
%	
% For details, please see: ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clf;

javaclasspath('-dynamic')                                                   % java configuration
filePath = '/media/thiago/ubuntu/datasets/imageDenoising/dl/';              % change to save generated data
ref = sprintf('%srefPatches.csv', filePath);                                % reference data
noisy = sprintf('%snoisyPatches.csv', filePath);                            % noisy data
L = 94500;                                                                  % number of observations
Ks = [[10, 5, 2]; [50, 10, 5]; [100, 20, 5]; [200, 20, 10]; [500, 50, 10]]; % dictionary's atoms (K << L)
N = 49; N1 = 7; N2 = 7;                                                     % features/variables/components
noIts = [10, 50, 100];                                                      % number of iterations
sparsities = [2, 3, 5, 7, 10];                                              % sparsity degree
solver = 'javaORMP';                                                        % solver for sparse approximation
methodChar = 'H';                                                           % method for RLS-DLA

for k=1:size(Ks, 1);
    for i=1:size(sparsities, 2);
        for j=1:size(noIts, 2);            
            K = Ks(k,1);
            K1 = Ks(k,2);
            K2 = Ks(k,3);
            sparsity = sparsities(i);
            noIt = noIts(i);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% MOD dictionary learning from estimated dictionary and reference data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODRef', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODRef', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(ref);
                X = X.';
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

                % Learning
                [A_hat,S_hat] = modDL(noIt, X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');    
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('MOD-Time: %f\n\n', toc);

            %% MOD dictionary learning from learned dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODRefNoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODRefNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(noisy);
                X = X.';
                A_hat = A_hat.';

                % Learning
                [A_hat,S_hat] = modDL(noIt, X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('MOD-Time: %f\n\n', toc);

            %% MOD dictionary learning from estimated dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODNoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

                % Learning
                [A_hat,S_hat] = modDL(noIt, X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';'); 
            end
            fprintf('MOD-Time: %f\n\n', toc);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% RLS-DLA dictionary learning from estimated dictionary and reference data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLARef', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLARef', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(ref);
                X = X.';
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
                metPar = cell(1,1);
                metPar{1} = struct('lamM', methodChar, 'lam0', 0.99, 'a', 0.95);
                if (strcmpi(methodChar,'E'));
                    metPar{1}.a = 0.15;
                end;
                if (strcmpi(methodChar,'H'));
                    metPar{1}.a = 0.10;
                end;

                % Learning
                A_hat = rlsdla(L, noIt, N, K, X, metPar, A_hat, sparsity);
                S_hat = sparseapprox(X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('RLS-DLA-Time: %f\n\n', toc);

            %% RLS-DLA dictionary learning from learned dictionary and noisy data
            tic;
            dicFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLARefNoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLARefNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(noisy);
                X = X.';
                A_hat = A_hat.';

                % Learning
                A_hat = rlsdla(L, noIt, N, K, X, metPar, A_hat, sparsity);
                S_hat = sparseapprox(X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dicFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('RLS-DLA-Time: %f\n\n', toc);

            %% RLS-DLA dictionary learning from estimated dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLANoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLANoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

                % Learning
                A_hat = rlsdla(L, noIt, N, K, X, metPar, A_hat, sparsity);
                S_hat = sparseapprox(X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('RLS-DLA-Time: %f\n\n', toc);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% K-SVD dictionary learning from estimated dictionary and reference data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDRef', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDRef', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(ref);
                X = X.';
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

                % Learning
                [A_hat,S_hat] = ksvd(noIt, X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('K-SVD-Time: %f\n\n', toc);

            %% K-SVD dictionary learning from learned dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDRefNoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDRefNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(noisy);
                X = X.';
                A_hat = A_hat.';

                % Learning
                [A_hat,S_hat] = ksvd(noIt, X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('K-SVD-Time: %f\n\n', toc);

            %% K-SVD dictionary learning from estimated dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDNoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

                % Learning
                [A_hat,S_hat] = ksvd(noIt, X, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('K-SVD-Time: %f\n\n', toc);


%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %% K-HOSVD dictionary learning from estimated dictionary and reference data
%             tic;
%             dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDRef', L, K, noIt, solver, sparsity);
%             sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDRef', L, K, noIt, solver, sparsity);
%             fprintf('%s\n', dictFileName);
%             fprintf('%s\n', sparseFileName);
%             if ~exist(dictFileName,'file')
%                 X = dlmread(ref);
%                 X = X.';
%                 A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
% 
%                 % Learning
%                 [A_hat,S_hat] = khosvd(noIt, X, A_hat, N1, N2, solver, 'tnz', sparsity);
%                 A_hat = A_hat.';
%                 S_hat = S_hat.';
% 
%                 % Saving
%                 dlmwrite(dictFileName, A_hat, 'delimiter', ';');
%                 dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
%             end
%             fprintf('K-HOSVD-Time: %f\n\n', toc);
% 
%             %% K-HOSVD dictionary learning from learned dictionary and noisy data
%             tic;
%             dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDRefNoisy', L, K, noIt, solver, sparsity);
%             sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDRefNoisy', L, K, noIt, solver, sparsity);
%             fprintf('%s\n', dictFileName);
%             fprintf('%s\n', sparseFileName);
%             if ~exist(dictFileName,'file')
%                 X = dlmread(noisy);
%                 X = X.';
%                 A_hat = A_hat.';
% 
%                 % Learning
%                 [A_hat,S_hat] = khosvd(noIt, X, A_hat, N1, N2, solver, 'tnz', sparsity);
%                 A_hat = A_hat.';
%                 S_hat = S_hat.';
% 
%                 % Saving
%                 dlmwrite(dictFileName, A_hat, 'delimiter', ';');
%                 dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
%             end
%             fprintf('K-HOSVD-Time: %f\n\n', toc);
% 
%             %% K-HOSVD dictionary learning from estimated dictionary and noisy data
%             tic;
%             dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDNoisy', L, K, noIt, solver, sparsity);
%             sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDNoisy', L, K, noIt, solver, sparsity);
%             fprintf('%s\n', dictFileName);
%             fprintf('%s\n', sparseFileName);
%             if ~exist(dictFileName,'file')
%             A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
% 
%             % Learning
%             [A_hat,S_hat] = khosvd(noIt, X, A_hat, N1, N2, solver, 'tnz', sparsity);
%             A_hat = A_hat.';
%             S_hat = S_hat.';
% 
%             % Saving
%             dlmwrite(dictFileName, A_hat, 'delimiter', ';');
%             dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
%             end
%             fprintf('K-HOSVD-Time: %f\n\n', toc);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% T-MOD dictionary learning from estimated dictionary and reference data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODRef', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODRef', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(ref);
                X = X.';
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
                [A_hat1, A_hat2] = krondecomp(A_hat, N1, N2, K1, K2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*

                % Learning
                [A_hat,S_hat] = tmod(noIt, X(:, 1:3000), A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('T-MOD-Time: %f\n\n', toc);

            %% T-MOD dictionary learning from learned dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODRefNoisy', L, K, noIt, solver, sparsity);
            sparsefileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODRefNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                X = dlmread(noisy);
                X = X.';
                A_hat = A_hat.';
                [A_hat1, A_hat2] = krondecomp(A_hat, N1, N2, K1, K2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*

                % Learning
                [A_hat,S_hat] = tmod(noIt, X(:, 1:3000), A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('T-MOD-Time: %f\n\n', toc);

            %% T-MOD dictionary learning from estimated dictionary and noisy data
            tic;
            dictFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODNoisy', L, K, noIt, solver, sparsity);
            sparseFileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODNoisy', L, K, noIt, solver, sparsity);
            fprintf('%s\n', dictFileName);
            fprintf('%s\n', sparseFileName);
            if ~exist(dictFileName,'file')
                A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
                [A_hat1, A_hat2] = krondecomp(A_hat, N1, N2, K1, K2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*

                % Learning
                [A_hat,S_hat] = tmod(noIt, X(:, 1:3000), A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFileName, A_hat, 'delimiter', ';');
                dlmwrite(sparseFileName, S_hat, 'delimiter', ';');
            end
            fprintf('T-MOD-Time: %f\n\n', toc);
        end
    end
end
