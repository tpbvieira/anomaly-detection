%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project to evaluate Dictionary Learning methods for image       %%%
%%% denoising                                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: Thiago Vieira, UnB, 2017                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% General goal: ...
%
% For details, please see:
% TENSOR-BASED ALGORITHMS FOR LEARNING MULTIDIMENSIONAL SEPARABLE 
% DICTIONARIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clf;

javaclasspath('-dynamic')                                                   % java configuration
filePath = '/media/thiago/ubuntu/datasets/imageDenoising/dl/';              % mudar para o local onde deseja guardar os arquivos
L = 94500;                                                                  % observações utilizadas no aprendizado
K = 100; N1 = 20; N2 = 5;                                                   % atoms do dicionário
N = 49; M1 = 7; M2 = 7;                                                     % features/variáveis/componentes
noIt = 1;                                                                   % iterações para aprendizado
solver = 'javaORMP';                                                        % mais rápido que OMP
sparsity = 2;                                                               % grau de espasidade ( quantidade desejada de valores não-zedo por coluna )
methodChar = 'H';                                                           % method for RLS-DLA


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MOD dictionary learning from estimated dictionary and reference data
tic;
X = dlmread(sprintf('%srefPatches.csv', filePath),';');
X = X.';
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
[A_hat,S_hat] = modDL(noIt, X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nMOD-Time: %f', toc);

%% MOD dictionary learning from learned dictionary and noisy data
tic;
X = dlmread(sprintf('%snoisyPatches.csv', filePath),';');
X = X.';
A_hat = A_hat.';

% Learning
[A_hat,S_hat] = modDL(noIt, X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nMOD-Time: %f', toc);

%% MOD dictionary learning from estimated dictionary and noisy data
tic;
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
[A_hat,S_hat] = modDL(noIt, X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nMOD-Time: %f', toc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RLS-DLA dictionary learning from estimated dictionary and reference data
tic;
X = dlmread(sprintf('%srefPatches.csv', filePath),';');
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
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLARef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLARef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nRLS-DLA-Time: %f', toc);

%% RLS-DLA dictionary learning from learned dictionary and noisy data
tic;
X = dlmread(sprintf('%snoisyPatches.csv', filePath),';');
X = X.';
A_hat = A_hat.';

% Learning
A_hat = rlsdla(L, noIt, N, K, X, metPar, A_hat, sparsity);
S_hat = sparseapprox(X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLARefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLARefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nRLS-DLA-Time: %f', toc);

%% RLS-DLA dictionary learning from estimated dictionary and noisy data
tic;
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
A_hat = rlsdla(L, noIt, N, K, X, metPar, A_hat, sparsity);
S_hat = sparseapprox(X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLANoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLANoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nRLS-DLA-Time: %f', toc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% K-SVD dictionary learning from estimated dictionary and reference data
tic;
X = dlmread(sprintf('%srefPatches.csv', filePath),';');
X = X.';
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
[A_hat,S_hat] = ksvd(noIt, X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nK-SVD-Time: %f', toc);

%% K-SVD dictionary learning from learned dictionary and noisy data
tic;
X = dlmread(sprintf('%snoisyPatches.csv', filePath),';');
X = X.';
A_hat = A_hat.';

% Learning
[A_hat,S_hat] = ksvd(noIt, X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nK-SVD-Time: %f', toc);

%% K-SVD dictionary learning from estimated dictionary and noisy data
tic;
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
[A_hat,S_hat] = ksvd(noIt, X, A_hat, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nK-SVD-Time: %f', toc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% K-HOSVD dictionary learning from estimated dictionary and reference data
tic;
X = dlmread(sprintf('%srefPatches.csv', filePath),';');
X = X.';
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
[A_hat,S_hat] = khosvd(noIt, X, A_hat, M1, M2, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nK-HOSVD-Time: %f', toc);

%% K-HOSVD dictionary learning from learned dictionary and noisy data
tic;
X = dlmread(sprintf('%snoisyPatches.csv', filePath),';');
X = X.';
A_hat = A_hat.';

% Learning
[A_hat,S_hat] = khosvd(noIt, X, A_hat, M1, M2, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nK-HOSVD-Time: %f', toc);

%% K-HOSVD dictionary learning from estimated dictionary and noisy data
tic;
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );

% Learning
[A_hat,S_hat] = khosvd(noIt, X, A_hat, M1, M2, solver, 'tnz', sparsity);
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nK-HOSVD-Time: %f', toc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% T-MOD dictionary learning from estimated dictionary and reference data
tic;
X = dlmread(sprintf('%srefPatches.csv', filePath),';');
X = X.';
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
[A_hat1, A_hat2] = krondecomp(A_hat, M1, M2, N1, N2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*

% Learning
[A_hat,S_hat] = tmod(noIt, X(:, 1:3000), A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODRef', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nT-MOD-Time: %f', toc);

%% T-MOD dictionary learning from learned dictionary and noisy data
tic;
X = dlmread(sprintf('%snoisyPatches.csv', filePath),';');
X = X.';
A_hat = A_hat.';
[A_hat1, A_hat2] = krondecomp(A_hat, M1, M2, N1, N2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*

% Learning
[A_hat,S_hat] = tmod(noIt, X(:, 1:3000), A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODRefNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nT-MOD-Time: %f', toc);

%% T-MOD dictionary learning from estimated dictionary and noisy data
tic;
A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
[A_hat1, A_hat2] = krondecomp(A_hat, M1, M2, N1, N2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*

% Learning
[A_hat,S_hat] = tmod(noIt, X(:, 1:3000), A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
A_hat = A_hat.';
S_hat = S_hat.';

% Saving
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, A_hat, 'delimiter', ';');
fileName = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODNoisy', L, K, noIt, solver, sparsity);
dlmwrite(fileName, S_hat, 'delimiter', ';');
fprintf('\nT-MOD-Time: %f', toc);