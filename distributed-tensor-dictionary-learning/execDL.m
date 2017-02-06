function res = execDL(L, N, K, snr, methodChar, s, noIt, nofTrials, makeFig)
    % executed a selected dictionary learning algorithm to recover a known 
    % dictionary, Random Gaussian dictionary, randomly generated (training)
    % data with added Gaussian noise.
    %
    % The result (res) is stored in a structure. If this file exist the
    % results of the new trials are added to the previous results.
    %
    % To generate new data, make sure to delete or rename 'ex210xsyynn.mat'.
    %
    % This code is a version of Karl Skretting work.
    %-------------------------------------------------------------------------
    % parameters:
    %   res         = a struct which is also stored in 'ex210xsyynn.mat'
    %   L           = number of training vectors to use
    %   N           = number of lines of the file where the set of training vectors, X, is stored during design
    %   K           = number of columns of the file where the set of training vectors, X, is stored during design
    %   snr         = snr for added noise
    %   methodChar  = the method to use, (x above: K I L Q C H E A M)
    %                   'K' = K-SVD, 
    %                   'A' = AK-SVD,
    %                   'T' = K-HOSVD,
    %                   'D' = MOD,
    %                   'O' = T-MOD,
    %                   'M' = ILS-DLA MOD,
    %                   'I' = ILS-DLA MOD (java),
    %                   'B' = RLS-DLA miniBatch
    %                   'L', 'Q', 'C', 'H' or 'E' = RLS-DLA (java),
    %   s           = sparseness, number of non-zero coefficients, default 5 
    %   noIt        = number of iterations to do for each trial, default 200
    %   nofTrials   = number of trials to do, default 1
    %   makeFig     = 0 (false)/1 (true), default 1
    %-------------------------------------------------------------------------    
    % Exemples:
    % res = ex210(L, snr, methodChar, s, noIt, nofTrials, makeFig);
    % res = ex210(2000, 20, 'M', 5, 100, 10, 1); % 10 new trials
    % res = ex210(2000, 20, 'M', 5, 100, 0, 1);  % no new trials, just plot stored res
    % res = ex210('many');                  % special case, see (and edit) code
    %----------------------------------------------------------------------

    mfile = 'execDL';

    % validate the number of arguments
    if (nargin < 3)
       error([mfile,': wrong number of arguments, see help.']);
    end
    if (nargin < 4); s = 5; end;
    if (nargin < 5); noIt = 200; end;
    if (nargin < 6); nofTrials = 1; end;
    if (nargin < 7); makeFig = 1; end;

    if (strcmpi(methodChar,'K'))                                                % 'K' = K-SVD,   
        method = 'K-SVD';
    elseif (strcmpi(methodChar,'A'))                                            % 'A' = AK-SVD,
        method = 'AK-SVD';    
    elseif (strcmpi(methodChar,'T'))                                            % 'T' = K-HOSVD
        method = 'K-HOSVD';
    elseif (strcmpi(methodChar,'D'))                                            % 'D' = MOD,
        method = 'MOD';
    elseif (strcmpi(methodChar,'O'))                                            % 'O' = T-MOD,
        method = 'T-MOD';
    elseif (strcmpi(methodChar,'M'))                                            % 'M' = ILS-DLA MOD,
        method = 'ILS-DLA MOD';
    elseif (strcmpi(methodChar,'I'))                                            % 'I' = ILS-DLA MOD (java),
        method = 'ILS-DLA MOD (Java)';    
    elseif (strcmpi(methodChar,'B'))                                            % 'B' = RLS-DLA miniBatch
        method = 'RLS-MiniBatch';
    else                                                                        % 'L', 'Q', 'C', 'H' or 'E' = RLS-DLA (java),
        method = ['RLS-DLA (Java)',methodChar];
    end

    % parameters
    metPar = cell(1,1);
    metPar{1} = struct('lamM',methodChar,'lam0',0.99,'a',0.95);
    if (strcmpi(methodChar,'E')); metPar{1}.a = 0.15; end;
    if (strcmpi(methodChar,'H')); metPar{1}.a = 0.10; end;
    betalim = 8.11;                                                             % 1 - d'*dorg < 0.01  ==> |cos(beta)|>0.01
    
    % java configuration
    javaclasspath('-dynamic')

    % outputs
    ResultFile = [mfile, methodChar, sprintf('%1i%02i%02i.mat', s, floor(L/1000), floor(snr))];
    res = struct('beta', zeros(K, nofTrials), ...
                 'nofTrials', nofTrials, ...
                 's',s, ...
                 'N',N, ...
                 'K',K, ...
                 'noIt', noIt, ...
                 'L', L, ...
                 'snr', snr, ...
                 'methodChar', methodChar, ...
                 'metPar', metPar, ...
                 'method', method);

    % load previous files and verify if the execution can be avoided
    if exist(ResultFile,'file')
        oldFile = dir(ResultFile);
        disp([mfile,': add to results stored in ', ResultFile,', (created ', oldFile.date,').']);
        load(ResultFile);
        trialsDone = res.nofTrials;
        if (nofTrials > 0)
            res.beta = [res.beta, zeros(K, nofTrials)];
            res.nofTrials = res.nofTrials + nofTrials;
        end
    else
        trialsDone = 0;
    end

    % logging
    disp(' ');
    disp([mfile,': noIt=',int2str(noIt),', nofTrials=',int2str(nofTrials)]);
    
    timestart = now();
    for trial = 1:nofTrials
        % logging
        disp(' ');
        disp([mfile,': ',method,' L=',int2str(L),' snr=',num2str(snr), ...
            ', s=',int2str(s), ...
            '. Do trial number ',int2str(trial),' of ',int2str(nofTrials),...
            ', each using ',int2str(noIt),' iterations.']);

        % data generation
        A = dictmake(N, K, 'G');                                                % Generate a dictionary
        X = datamake(A, L, s, snr, 'G');                                        % Generate a random (learning) data set using a given dictionary
        A_hat = dictnormalize( X(:,floor(0.85 * L - K) + (1:K)) );              % Normalize and arrange the vectors of a initial estimated dictionary 
        [A_hat1, A_hat2] = krondecomp(A_hat, 4, 5, 5, 10);
    	A_hat = kron(A_hat1, A_hat2);
        tic;

        if (strcmpi(methodChar,'K'))                                            % K-SVD
            A_hat = ksvd(noIt, K, X, A_hat, 'javaORMP', 'tnz',s);
        elseif(strcmpi(methodChar,'A'))                                         % AK-SVD
            A_hat = aksvd(noIt, K, X, A_hat, 'javaORMP', 'tnz',s);
        elseif(strcmpi(methodChar,'T'))                                         % K-HOSVD
            A_hat = khosvd2(noIt, X, A_hat, 5, 4, 'javaORMP', 'tnz',s);
        elseif strcmpi(methodChar,'D')                                          % MOD
            A_hat = modDL(noIt, X, A_hat, 'javaORMP', 'tnz',s);
        elseif (strcmpi(methodChar,'O'))                                        % T-MOD
            A_hat = tmod(noIt, X, A_hat, A_hat1, A_hat2, 'javaORMP', 'tnz',s);
        elseif strcmpi(methodChar,'M')                                          % ILS-DLA MOD
            A_hat = ilsdla(noIt, X, A_hat, 'javaORMP', 'tnz',s);
        elseif (strcmpi(methodChar,'I'))                                        % ILS-DLA MOD (java)           
            A_hat = ilsdlajava(noIt, N, K, X, A_hat, s);
        elseif (strcmpi(methodChar,'B'))                                        % MiniBatch
            mb = [1,25; 1,50; 1,125; 1,300];                                    % building block in minibatch
            v2p = sum( mb(:,1).*mb(:,2) );                                      % vectors to process (500)
            mb = repmat([ceil((L*noIt)/(v2p)),1],4,1).*mb;                      % we want: v2p >= (L*noIt)
            MBopt = struct('K',K, ...
                       'samet','mexomp', ...
                       'saopt',struct('tnz',s, 'verbose',0), ...
                       'minibatch', mb, ...
                       'lam0', 0.99, 'lam1', 0.9, ...
                       'PropertiesToCheck', {{}}, ...
                       'checkrate', L, ...
                       'verbose',0 );
            res.MBopt = MBopt;
            res.Ds = rlsdlaminibatch('X',X, MBopt);
            A_hat = res.Ds.D;
        else                                                                    % RLS-DLA (java)
            A_hat = rlsdla(L, noIt, N, K, X, metPar, A_hat, s);
        end

        t = toc;

        % compare the trained dictionary to the true dictionary
        beta = dictdiff(A_hat, A, 'all-1', 'thabs');
        beta = beta*180/pi;                                                     % degrees
        disp(['Trial ',int2str(trial), ...
            sprintf(': %.2f seconds used.',t), ...
            ' Indentified ',int2str(sum(beta<betalim)), ...
            ' atoms out of ',int2str(K), ...
            '. Mean angle is ',num2str(mean(beta)),' degrees.']);

        res.beta(:,trialsDone + trial) = beta(:);
        %
        timeleft = (now()-timestart)*((nofTrials - trial) / trial);
        disp(['Estimated finish time is ',datestr(now()+timeleft)]);
        %
        if (trial == nofTrials)
            save(ResultFile, 'res' );
        end
    end

    %  make a plot
    if makeFig 
        betatab = [0.25:0.25:10, 10.5:0.5:25];
        y1 = zeros(numel(betatab),1);
        for i=1:numel(betatab);
            y1(i) = sum(res.beta(:) < betatab(i));
        end
        % overwrite current figure
        clf; hold on; grid on;
        plot(betatab,y1/res.nofTrials);
        p = 100*nnz(res.beta <= betalim)/numel(res.beta);                       % percent identified
        % write text from bottom and up
        x = 9.5; y = 2.5; dy = 3;
        h = text(x, y, ['Percent identified (\beta_l_i_m ',sprintf('= %5.2f) : %6.2f',betalim,p)]);
        set(h,'BackgroundColor',[1,1,1]); y = y+dy;
        h = text(x, y,['Noise in data has snr of ',num2str(res.snr),' dB']);
        set(h,'BackgroundColor',[1,1,1]); y = y+dy;
        h = text(x, y,['Number of training vectors L = ',int2str(res.L)]);
        set(h,'BackgroundColor',[1,1,1]); y = y+dy;
        h = text(x, y,['Training with ',int2str(res.noIt),' iterations.']);
        set(h,'BackgroundColor',[1,1,1]); y = y+dy;
        h = text(x, y,['Dictionary learning method: ',res.method]);
        set(h,'BackgroundColor',[1,1,1]); 
        % write some points of the line
        x = 10.5;
        for y = 25:5:45
            I = find(y1 > y*res.nofTrials);    
            if numel(I)
                i = I(1);
                if ((i>1) && (y1(i) > y1(i-1)))
                    xp = betatab(i-1) + (betatab(i)-betatab(i-1))*(y*res.nofTrials-y1(i-1))/(y1(i)-y1(i-1));
                else
                    xp = betatab(i);
                end
                if (xp < 10)
                    h = text(x, y-1, [num2str(y),' identified for \beta_l_i_m = ',sprintf('%2.2f',xp),'.']);
                    set(h,'BackgroundColor',[1,1,1]);
                end
            end
        end
        %
        title({['Nof atoms identified, average over ',int2str(res.nofTrials),' trials, s=',int2str(s),'.'];
               ['Plot generated ',datestr(now()),'. (',mfile,'.m ver 1.1, by Karl Skretting, UiS).']} );
        xlabel('Limit for positive identification, \beta_l_i_m [degrees].');
        ylabel(['Number of identified atoms of ',int2str(K)]);
        print( gcf, '-depsc2', [ResultFile(1:(end-4)),'.eps'] );
        disp(['\nPrinted figure as: ',[ResultFile(1:(end-4)),'.eps']]);
    end

    return