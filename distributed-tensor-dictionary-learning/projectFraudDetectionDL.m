%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Evaluates Dictionary Learning methods for fraud detection       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: Thiago Vieira, UnB, 2017                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% General goal: Evaluates Dictionary Learning methods for fraud detection
% and saves the generated data into files to be reused. Initially it is
% verified if the target parameter combination was previously evaluated, in
% order to perform new evaluation only for new parameter combinations. The
% following dictionary learning methods are evaluated:
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
filePath = '/media/thiago/ubuntu/datasets/fraudDetection/';                 % change to save generated data
trainDataPath = sprintf('%stest_under_data.csv', filePath);                 % train data

fid = fopen(trainDataPath,'r');
HL = 1;     %for example
HC = 1;     %in your case
NF = 7;     %1000 desired fields
lineformat = [repmat('%*s',1,HC) repmat('%f',1,NF)];
train_data = cell2mat(textscan(fid, lineformat, 'HeaderLines', HL, 'Delimiter', ','));
fclose (fid);
train_data = train_data.';
L = size(train_data, 2);                                                                    % number of observations
% Ks = [[10, 5, 2]; [50, 10, 5]; [100, 20, 5]; [200, 20, 10]; [500, 50, 10]];               % dictionary's atoms (K << L)
Ks = [[10, 5, 2]; [50, 10, 5]; [100, 20, 5]];                                               % dictionary's atoms (K << L)
N = 7; N1 = 7; N2 = 1;                                                                      % features/variables/components
sparsities = [2, 5];                                                                        % sparsity degrees
noIts = [10, 50, 100];                                                                      % number of iterations
solver = 'javaORMP';                                                                        % solver for sparse approximation
methodChar = 'H';                                                                           % method for RLS-DLA

%% Dictionary Learning Cross-Validation for fraud detection data
fprintf('## Dictionary Learning from fraud detection data');
for k=1:size(Ks, 1);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    for i=1:size(sparsities, 2);
        for j=1:size(noIts, 2);
            %% Settings
            K = Ks(k,1);
            K1 = Ks(k,2);
            K2 = Ks(k,3);
            sparsity = sparsities(i);
            noIt = noIts(j);
            fprintf('\n%s', sprintf('\n## L=%i_K=%i_noIt=%i_solver=%s_tnz=%i', L, K, noIt, solver, sparsity));
            
            %% MOD dictionary learning from estimated dictionary and noisy data
            tic;
            dictFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictMODNoisy', L, K, noIt, solver, sparsity);
            sparseFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeMODNoisy', L, K, noIt, solver, sparsity);
            if ~exist(dictFilePath,'file')                
                A_hat = dictnormalize( train_data(:, floor(0.85 * L - K) + (1:K)) );
                A_hat = single(A_hat);
                
                % Learning
                [A_hat,S_hat] = modDL(noIt, train_data, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFilePath, A_hat, 'delimiter', ';');
                dlmwrite(sparseFilePath, S_hat, 'delimiter', ';'); 
            end
            fprintf('\nMOD-Time: %f', toc);
            
            %% RLS-DLA dictionary learning from estimated dictionary and noisy data
            tic;
            dictFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictRLS-DLANoisy', L, K, noIt, solver, sparsity);
            sparseFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeRLS-DLANoisy', L, K, noIt, solver, sparsity);
            if ~exist(dictFilePath,'file')
                A_hat = dictnormalize( train_data(:, floor(0.85 * L - K) + (1:K)) );
                A_hat = single(A_hat);
                
                metPar = cell(1,1);
                metPar{1} = struct('lamM', methodChar, 'lam0', 0.99, 'a', 0.95);
                if (strcmpi(methodChar,'E'));
                    metPar{1}.a = 0.15;
                end;
                if (strcmpi(methodChar,'H'));
                    metPar{1}.a = 0.10;
                end;

                % Learning
                A_hat = rlsdla(L, noIt, N, K, train_data, metPar, A_hat, sparsity);
                S_hat = sparseapprox(train_data, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFilePath, A_hat, 'delimiter', ';');
                dlmwrite(sparseFilePath, S_hat, 'delimiter', ';');
            end
            fprintf('\tRLS-DLA-Time: %f', toc);

            %% K-SVD dictionary learning from estimated dictionary and noisy data
            tic;
            dictFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-SVDNoisy', L, K, noIt, solver, sparsity);
            sparseFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-SVDNoisy', L, K, noIt, solver, sparsity);
            if ~exist(dictFilePath,'file')
                A_hat = dictnormalize( train_data(:, floor(0.85 * L - K) + (1:K)) );
                
                % Learning
                [A_hat,S_hat] = ksvd(noIt, train_data, A_hat, solver, 'tnz', sparsity);
                A_hat = A_hat.';
                S_hat = S_hat.';

                % Saving
                dlmwrite(dictFilePath, A_hat, 'delimiter', ';');
                dlmwrite(sparseFilePath, S_hat, 'delimiter', ';');
            end
            fprintf('\tK-SVD-Time: %f', toc);

%             %% K-HOSVD dictionary learning from estimated dictionary and noisy data
%             tic;
%             dictFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictK-HOSVDNoisy', L, K, noIt, solver, sparsity);
%             sparseFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeK-HOSVDNoisy', L, K, noIt, solver, sparsity);
%             if ~exist(dictFilePath,'file')
%             A_hat = dictnormalize( train_data(:, floor(0.85 * L - K) + (1:K)) );
%             A_hat = single(A_hat);
% 
%             % Learning
%             [A_hat,S_hat] = khosvd(noIt, train_data, A_hat, N1, N2, solver, 'tnz', sparsity);
%             A_hat = A_hat.';
%             S_hat = S_hat.';
% 
%             % Saving
%             dlmwrite(dictFilePath, A_hat, 'delimiter', ';');
%             dlmwrite(sparseFilePath, S_hat, 'delimiter', ';');
%             end
%             fprintf('\tK-HOSVDNoisy-Time: %f', toc);
            
%             %% T-MOD dictionary learning from estimated dictionary and noisy data
%             tic;
%             dictFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'dictT-MODNoisy', L, K, noIt, solver, sparsity);
%             sparseFilePath = sprintf('%s%s_L=%i_K=%i_noIt=%i_solver=%s_tnz=%i.csv', filePath, 'sparseCodeT-MODNoisy', L, K, noIt, solver, sparsity);
%             if ~exist(dictFilePath,'file')
%                 A_hat = dictnormalize( X(:, floor(0.85 * L - K) + (1:K)) );
%                 [A_hat1, A_hat2] = krondecomp(A_hat, N1, N2, K1, K2);                       % Make the data separable decomposing the approximation of A_hat and generating new A_hat*
%                 A_hat1 = single(A_hat1);
%                 A_hat2 = single(A_hat2);
%                 
%                 % Learning
%                 [A_hat,S_hat] = tmod(noIt, X, A_hat1, A_hat2, solver, 'tnz', sparsity);  % FIXME: Foi definido o valor de 3000 amostras de X para o treimaneto.
%                 A_hat = A_hat.';
%                 S_hat = S_hat.';
% 
%                 % Saving
%                 dlmwrite(dictFilePath, A_hat, 'delimiter', ';');
%                 dlmwrite(sparseFilePath, S_hat, 'delimiter', ';');
%             end
%             fprintf('\tT-MOD-Time: %f', toc);
        end
    end
end
