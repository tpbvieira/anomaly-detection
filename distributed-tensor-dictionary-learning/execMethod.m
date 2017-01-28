function res = execMethod(L, snr, methodChar, s, noIt, nofTrials, makeFig)
    % ex210           Experiment is to recover a known dictionary, size 20x50
    %                 Random Gaussian dictionary, randomly generated (training) 
    % data with added Gaussian noise.
    %
    % The result (res) is stored in a mat-file, if this file exist the results
    % of the new trials are added to the previous results.
    % To generate new data, make sure to delete or rename 'ex210xsyynn.mat'.
    %   x  is  'K' for K-SVD, 
    %          'I' for ILS-DLA (MOD java implementation),  
    %      and 'L', 'Q', 'C', 'H' or 'E' for RLS-DLA  (java impl.)
    %      and 'A'  (Approximate K-SVD)
    %      and 'M'  (MOD, matlab implementation)
    %      and 'B'  (miniBatch implementation of RLS-DLA)
    %   s  is sparseness s
    %   yy is floor(L/1000) where L is number of training vectors
    %   nn is floor(snr) for added noise level
    % Also a figure may be generated and saved as eps-file.
    %
    % res = ex210(L, snr, methodChar, s, noIt, nofTrials, makeFig);
    % res = ex210(2000, 20, 'M', 5, 100, 10, 1); % 10 new trials
    % res = ex210(2000, 20, 'M', 5, 100, 0, 1);  % no new trials, just plot stored res
    % res = ex210('many');                  % special case, see (and edit) code
    %-------------------------------------------------------------------------
    % arguments:
    %   res        a struct which is also stored in 'ex210xsyynn.mat'
    %   L          number of training vectors to use
    %   snr        snr for added noise
    %   methodChar    the method to use, (x above: K I L Q C H E A M)
    %   s          sparseness, number of non-zero coefficients, default 5 
    %   noIt       number of iterations to do for each trial, default 200
    %   nofTrials  number of trials to do, default 1
    %   makeFig    0/1, default 1
    %-------------------------------------------------------------------------

    %----------------------------------------------------------------------
    % Copyright (c) 2009.  Karl Skretting.  All rights reserved.
    % University of Stavanger (Stavanger University), Signal Processing Group
    % Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
    % 
    % HISTORY:  dd.mm.yyyy
    % Ver. 1.0  06.08.2009  KS: made file
    % Ver. 1.1  12.04.2013  KS: some extensions, s as argument, more methods
    %----------------------------------------------------------------------

    mfile = 'funcions';

    % check the number of arguments
    if (nargin < 3)
       error([mfile,': wrong number of arguments, see help.']);
    end
    if (nargin < 4); s = 5; end;
    if (nargin < 5); noIt = 200; end;
    if (nargin < 6); nofTrials = 1; end;
    if (nargin < 7); makeFig = 1; end;

    % the file where the set of training vectors, X, is stored during design
    N = 20;
    K = 50;

    if (strcmpi(methodChar,'A'))   
        method = 'AK-SVD';    
    elseif (strcmpi(methodChar,'B'))   
        method = 'RLS-MiniBatch';    
    elseif (strcmpi(methodChar,'K'))   
        method = 'K-SVD';    
    elseif (strcmpi(methodChar,'I'))   
        method = 'ILS-DLA';                                                     % I = MOD java impl.
    elseif (strcmpi(methodChar,'M'))   
        method = 'MOD';                                                         % M = MOD matlab impl.
    else
        method = ['RLS-DLA ',methodChar];                                       % RLS-DLA with different approaches for increasing the forgetting factor
    end

    % misterious parameters
    metPar = cell(1,1);
    metPar{1} = struct('lamM',methodChar,'lam0',0.99,'a',0.95);
    if (strcmpi(methodChar,'E')); metPar{1}.a = 0.15; end;
    if (strcmpi(methodChar,'H')); metPar{1}.a = 0.10; end;
    betalim = 8.11;                                                             % 1 - d'*dorg < 0.01  ==> |cos(beta)|>0.01

    ResultFile = [mfile,methodChar,sprintf('%1i%02i%02i.mat',s,floor(L/1000),floor(snr))];

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

    if (methodChar == 'B')                                                      % minibatch; dictlearn_mb
        mb = [1,25; 1,50; 1,125; 1,300];                                        % building block in minibatch
        v2p = sum( mb(:,1).*mb(:,2) );                                          % vectors to process (500)
        mb = repmat([ceil((L*noIt)/(v2p)),1],4,1).*mb;  % we want: v2p >= (L*noIt)

        MBopt = struct('K',K, ...
                   'samet','mexomp', ...
                   'saopt',struct('tnz',s, 'verbose',0), ...
                   'minibatch', mb, ...
                   'lam0', 0.99, 'lam1', 0.9, ...
                   'PropertiesToCheck', {{}}, ...
                   'checkrate', L, ...
                   'verbose',0 );
        res.MBopt = MBopt;
    end

    if exist(ResultFile,'file')
        oldFile = dir(ResultFile);
        disp([mfile,': add to results stored in ', ResultFile,', (created ', oldFile.date,').']);
        load(ResultFile);    % load res
        trialsDone = res.nofTrials;
        if (nofTrials > 0)
            res.beta = [res.beta, zeros(K, nofTrials)];
            res.nofTrials = res.nofTrials + nofTrials;
            % the other fields should be unchanged
        end
    else
        trialsDone = 0;
    end

    disp(' ');
    disp([mfile,': noIt=',int2str(noIt),', nofTrials=',int2str(nofTrials)]);
    java_access;                                                                %test java availability

    timestart = now();
    for trial = 1:nofTrials
        % logging
        disp(' ');
        disp([mfile,': ',method,' L=',int2str(L),' snr=',num2str(snr), ...
            ', s=',int2str(s), ...
            '. Do trial number ',int2str(trial),' of ',int2str(nofTrials),...
            ', each using ',int2str(noIt),' iterations.']);

        Dorg = dictmake(N, K, 'G');                                             % Generate a dictionary
        X = datamake(Dorg, L, s, snr, 'G');                                     % Generate a random data set using a given dictionary
        % sumXX = sum(sum(X.*X));
        D = dictnormalize( X(:,floor(0.85*L-K)+(1:K)) );                        % Normalize and arrange the vectors of a dictionary 
        javaclasspath('-dynamic')

        tic;

        if strcmpi(methodChar,'M')                                              % MOD, matlab implementation
            for it = 1:noIt
                W = sparseapprox(X, D, 'javaORMP', 'tnz',s);
                D = (X*W')/(W*W');
                D = D ./ repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
            end
        elseif (strcmpi(methodChar,'K') || strcmpi(methodChar,'A'))             % *-SVD
            for it = 1:noIt
                W = sparseapprox(X, D, 'javaORMP', 'tnz',s);                    % find weights, using dictionary D
                R = X - D*W;
                if strcmpi(methodChar,'K')                                      % K-SVD
                    for k=1:K
                        I = find(W(k,:));
                        Ri = R(:,I) + D(:,k)*W(k,I);
                        [U,S,V] = svds(Ri,1,'L');
                        D(:,k) = U;
                        W(k,I) = S*V';
                        R(:,I) = Ri - D(:,k)*W(k,I);
                    end
                else                                                            % AK-SVD
                    for a = 1:3
                        for k=1:K
                            I = find(W(k,:));
                            Ri = R(:,I) + D(:,k)*W(k,I);
                            dk = Ri * W(k,I)';
                            dk = dk/sqrt(dk'*dk);  % normalize
                            D(:,k) = dk;
                            W(k,I) = dk'*Ri;
                            R(:,I) = Ri - D(:,k)*W(k,I);
                        end
                    end
                end
            end
        elseif (strcmpi(methodChar,'B'))                                        % MiniBatch
            res.Ds = dictlearn_mb('X',X, MBopt);
            D = res.Ds.D;
        elseif (strcmpi(methodChar,'I'))                                        % ILS (java)
            jD0 = mpv2.SimpleMatrix(D);
            jDicLea  = mpv2.DictionaryLearning(jD0, 1);
            jDicLea.setORMP(int32(s), 1e-6, 1e-6);
            jDicLea.ilsdla( X(:), noIt );
            % snrTab = jDicLea.getSnrTab();
            jD =  jDicLea.getDictionary();
            D = reshape(jD.getAll(), N, K);
        else                                                                    % RLS-DLA (java)
            jD0 = mpv2.SimpleMatrix(D);
            jDicLea  = mpv2.DictionaryLearning(jD0, 1);
            jDicLea.setORMP(int32(s), 1e-6, 1e-6);
            jDicLea.setLambda( metPar{1}.lamM, metPar{1}.lam0, 1.0, (noIt*L)*metPar{1}.a );
            jDicLea.rlsdla( X(:), noIt );
            % snrTab = jDicLea.getSnrTab();
            jD =  jDicLea.getDictionary();
            D = reshape(jD.getAll(), N, K);
        end

        t = toc;

        % compare the trained dictionary to the true dictionary
        beta = dictdiff(D, Dorg, 'all-1', 'thabs');
        beta = beta*180/pi;  % want this in degrees
        disp(['Trial ',int2str(trial),sprintf(': %.2f seconds used.',t), ...
            ' Indentified ',int2str(sum(beta<betalim)),' atoms out of ',int2str(K), ...
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
        p = 100*nnz(res.beta <= betalim)/numel(res.beta);  % percent identified
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
        disp(['Printed figure as: ',[ResultFile(1:(end-4)),'.eps']]);
    end

    return