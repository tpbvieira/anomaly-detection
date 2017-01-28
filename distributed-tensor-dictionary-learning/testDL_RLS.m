function res = testDL_RLS(param)
% testDL_RLS      Test Recursive Least Squares Dictionary Learning
%
% replace 'old' files testDL_RLS_javaORMP.m and testDL_RLS_mexOMP.m
%
% The Berkeley Segmentation Dataset is used, 
% the 3 possible training datasets (A, B and C) are thus (almost) infinite.
% The dictionaries will be of size NxK
% 
% The results are also saved in 'testDL_RLS_mmmddhhmm.mat'.
% Result can be displayed by: testDLplot(res)
% 
% res = testDL_RLS(param);
%----------------------------------------------------------------------
% The input argument param is simply a struct with appropriate fields:
%   dataSet      : 'A' 8x8patches grayscale (- DC) gives N=64, K=256
%      'B' 12x12x3 patches color (-3 DC) gives N=432, K=512
%      'c' 16x16 patches grayscale (- DC) gives N=256, K=1024
%   vecSel       : which method to use for vector selection when 
%      learning. Can be: 'javaOMP', 'javaORMP', 'mexOMP', 'mexLasso'
%      ('mexOMP' is the same as ORMP, and gives same results as 'javaORMP')
%      Note if testing is done, 'mexOMP' or 'javaORMP' is used
%   targetPSNR   : the target PSNR during training
%      actual PSNR will often be 1 to 1.5 dB above the targetPSNR
%   mainIt       : number of different dataset to use in learning, ex: 100
%   dataLength   : number of vectors in each dataset, ex: 10000
%   dataBatch    : the dataset will be divided into batches, here the
%      number of vectors in each batch is given. ex 500. Note that the same
%      dictionary will be used for the complete batch, thus the value
%      should be smaller in the first iterations, this can be done by
%      assigning an array to dataBatch, ex [20,50,100,200,200,250,500], and the
%      first entry is used in the first iteration and so on, the last entry
%      is used for the rest of the iterations. 
%      all entries of 'dataBatch' should be factors of 'dataLength'
%   lambda0      : The start value for the forgetting factor in the 
%      Search-then-Converge scheme. Appropriate: 1-1/(5*K), 1-1/(10*K)
%   useTestData  : true or false
%   verbose      : 0, 1 or 2 (for very verbose)
% fields which normally is not used but may be given in input:
%   seedD0, avg_w0, avg_w1, avg_w2, avg_x0, avg_x1, avg_x2, 
%   avg_r0, avg_r1, avg_r2, avg_rr, min_diagC, max_diagC
% The output argument is also a struct with results (for each iteration)
%----------------------------------------------------------------------
% examples: 
% res = testDL_RLS();
% param = struct('vecSel','mexLasso', ...
%                'mainIt',50, 'dataLength',20000, 'dataSet','A', ...
%                'dataBatch',[20,50,100,200,200,250,500], 'targetPSNR', 38, ...
%                'useTestData',true, 'verbose', 0, 'lambda0', 0.998, ...
%                'avg_w0',1, 'avg_w1',1, 'avg_r2',1, 'avg_rr',1  );
% res = testDL_RLS( param );  
% testDLplot( res );

%----------------------------------------------------------------------
% Karl Skretting. 
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  21.02.2011  KS: made function based on testDL_RLS_javaORMP
% Ver. 1.1  18.03.2011  KS: added lambda0 
%----------------------------------------------------------------------

mfile = 'testDL_RLS';
if exist('stopp.fil','file')  
    disp(['Delete ''stopp.fil'' to run ',mfile]);
    res = 0;
    return;
end

% fields with fixed values
param.mfile = mfile;
param.dataPeak = 255;
param.saveResInFile = true;
% fields with default values which may be given in input
if ~isfield(param,'method');       param.method        = 'RLS';      end;
if ~isfield(param,'dataSet');      param.dataSet       = 'A';        end;
if ~isfield(param,'vecSel');       param.vecSel        = 'javaORMP'; end;
if ~isfield(param,'targetPSNR');   param.targetPSNR    = 38;         end;
if ~isfield(param,'mainIt');       param.mainIt        = 10;         end;
if ~isfield(param,'dataLength');   param.dataLength    = 5120;       end;
if ~isfield(param,'dataBatch');    param.dataBatch     = 512;        end;
if ~isfield(param,'verbose');      param.verbose       = 0;          end;
if ~isfield(param,'useTestData');  param.useTestData   = false;      end;

if       (param.dataSet == 'A');   N = 8*8;      K = 256;  parXn = [8,8];    col = false;
elseif   (param.dataSet == 'B');   N = 12*12*3;  K = 512;  parXn = [12,12];  col = true;
elseif   (param.dataSet == 'C');   N = 16*16;    K = 1024; parXn = [16,16];  col = false; 
end
if ~isfield(param,'lambda0');      param.lambda0       = 1-1/(5*K);  end;

% fields/variables derived from the input parameters (param)
param.targetMSE = N * param.dataPeak^2 * 10^(-param.targetPSNR/10);
param.vectorsToProcess = param.mainIt * param.dataLength;

if strcmpi(param.vecSel(1:4), 'java')
    java_access();  % check access to java
    param.sparseNum = int32(floor(0.5*N));          % ORMP returns when ||w||_0 == param.sparseNum
    param.absErrorLimit = sqrt(param.targetMSE);    % ORMP returns when ||r||_2 <= param.absErrorLimit
end
if strcmpi(param.vecSel(1:3), 'mex')
    % correct common 'misspellings'
    if strcmpi(param.vecSel, 'mexORMP'); param.vecSel = 'mexOMP'; end;
    if strcmpi(param.vecSel, 'mexLARS'); param.vecSel = 'mexLasso'; end;
    if isunix()  % check access to SPAMSS 
        % THIS MUST BE CHANGED TO MATCH THE ACTUAL SYSTEM
        SPAMScatalog = '~/Matlab/SPAMS/release/mkl64/';
    else
        % THIS MUST BE CHANGED TO MATCH THE ACTUAL PC 
        SPAMScatalog = 'D:\Karl\DICT\matlab\SPAMS\release\win32\';
        t = version('-release');
        if (eval(t(1:4)) < 2009) || strcmpi(t,'2009a')
            res = 'Works only when Matlab version is greater or equal to 2009b.';
            disp(res);
            return
        end
    end
    if exist([param.vecSel,'.m'],'file')
        disp(['Access to ',param.vecSel,'.m']);
    elseif exist([SPAMScatalog,param.vecSel,'.m'],'file')
        addpath(SPAMScatalog);
    else
        res = ['Can not locate ',param.vecSel,' on this computer.'];
        disp(res);
        return
    end
    param.paramLasso = struct(...
        'mode',   1, ...            
        'lambda', param.targetMSE, ...
        'L',      floor(0.9*N)  );
    param.paramOMP = struct(...
        'eps',    param.targetMSE, ...
        'L',      floor(0.5*N)  );
end

% the parameters used in getXfromBSDS300
parX = struct('imCat','train', 'no', param.dataLength, ...
              'myim2colPar',struct('n',parXn, 'i',[5,5], 'a','none', 'v',0), ...
              'col',col, 'sm',true, 'norm1lim', 5);
%           
% initialize result struct
res = struct(  'startTime', datestr(now()), ...
               'endTime', datestr(now()), ...
               'totalTime', -1, ...
               'vectorsProcessed', 0, ...
               'computer', computer(), ...
               'inputParam', param, ...
               'parX', parX, ...
               'fbA', zeros(1, param.mainIt+1), ...  
               'fbB', zeros(1, param.mainIt+1), ...
               'betamin', zeros(1, param.mainIt+1), ...
               'betaavg', zeros(1, param.mainIt+1), ...
               'betamse', zeros(1, param.mainIt+1), ...
               'delta0', zeros(1, param.mainIt), ...  % ||D_i - D_0||_F
               'deltai', zeros(1, param.mainIt), ...  % ||D_i - D_{i-1}||_F
               'D', zeros(N,K) );                     % final dictioary
if isfield(param,'avg_w0'); res.avg_w0 = zeros(1, param.mainIt); end;
if isfield(param,'avg_w1'); res.avg_w1 = zeros(1, param.mainIt); end;
if isfield(param,'avg_w2'); res.avg_w2 = zeros(1, param.mainIt); end;
if isfield(param,'avg_x0'); res.avg_x0 = zeros(1, param.mainIt); end;
if isfield(param,'avg_x1'); res.avg_x1 = zeros(1, param.mainIt); end;
if isfield(param,'avg_x2'); res.avg_x2 = zeros(1, param.mainIt); end;
if isfield(param,'avg_r0'); res.avg_r0 = zeros(1, param.mainIt); end;
if isfield(param,'avg_r1'); res.avg_r1 = zeros(1, param.mainIt); end;
if isfield(param,'avg_r2'); res.avg_r2 = zeros(1, param.mainIt); end;
if isfield(param,'avg_rr'); res.avg_rr = zeros(1, param.mainIt); end;  
if isfield(param,'min_diagC'); res.min_diagC = zeros(1, param.mainIt+1); end;
if isfield(param,'max_diagC'); res.max_diagC = zeros(1, param.mainIt+1); end;

disp(' ');
disp(['---- ',mfile,' - ',param.method,' -  started ',res.startTime]);

%% initialize
tstart = tic;
if isfield(param,'seedD0');          
    RandStream.setDefaultStream(RandStream('mt19937ar','seed',param.seedD0));
end
%
if strcmpi(param.method, 'javaRLS')
    jD = mpv2.SimpleMatrix( getXfromBSDS300(parX, 'no',K) );
    jDL  = mpv2.DictionaryLearning(jD, true);  % use DD
    jDL.setLambda('Q', param.lambda0, 1.0, 0.9*param.dataLength*param.mainIt);
    if strcmpi(param.vecSel,'javaORMP') || strcmpi(param.vecSel,'ORMP')
        jDL.setORMP(int32(min(N/2,80)), 1e-6, param.absErrorLimit);
    else
        res.jDL.setOMP(int32(min(N/2,80)), 1e-6, param.absErrorLimit);
    end
    jDL.setLoggOn();
    jDL.setVerbose(0);
    jDL.setParamI1(16);
    %
    D = reshape(jDL.getDictionary.getAll(), N, K);
    D0 = D;
    res.D = D;
    %
else  % initial Dictionary, D0, D and res.D, and C
    D0 = getXfromBSDS300(parX, 'no',K);  % alternativly X0 here
    diagG = 1./sqrt(sum(D0.^2));   % size 1xK,  D0 = X0*G  (scale the columns)
    D0 = D0 .* repmat(diagG, [size(D0,1) 1]);
    if (param.verbose > 0)
        disp([mfile,': generated initial dictionary D0 of size ',int2str(size(D0,1)),...
            'x',int2str(size(D0,2)),', type is ',param.dataSet,'.']);
    end
    [n,k] = size(D0);
    if (n ~= N) || (k ~= K)
        disp(['Error since wanted size was ',int2str(N),'x',int2str(K),'.']);
        return
    end
    D = D0;
    C = diag(diagG.^2);  % X0 = D0*W = D0*inv(G), C = inv(W'*W) = G'*G
    C = C + mean(diagG)*eye(K);  % why this ?
    res.D = D0;
end

if strcmpi(param.vecSel(1:4), 'java')
    % initialize dictionary learning for java
    jD = mpv2.SimpleMatrix(D);
    jDD = mpv2.SymmetricMatrix(K, K);
    jDD.eqInnerProductMatrix(jD);
    jMP = mpv2.MatchingPursuit(jD, jDD);
    jMP.setNormalized();
end
it = 0;    % number of main (outer) iterations

if param.useTestData   % test data
    testDataFile = ['testDLdata',param.dataSet,'.mat']; 
    if exist(testDataFile,'file') 
        Xtest = load(testDataFile);
        if isfield(Xtest, 'X') && (size(Xtest.X,1) == N)
            Xtest = Xtest.X(:,1:4096);   % not all here!
            if isfield(param,'avg_x0'); res.avg_x0 = mean(sum(Xtest ~= 0)); end;
            if isfield(param,'avg_x1'); res.avg_x1 = mean(sum(abs(Xtest))); end;
            if isfield(param,'avg_x2'); res.avg_x2 = mean(sqrt(sum(Xtest.*Xtest))); end;
        else
            disp([mfile,': ',testDataFile,' does not contain correct test data.']);
            clear Xtest
            param.useTestData = false;
        end
    else
        disp([mfile,': does not find file for test data, ',testDataFile,...
            ', simly solved by not doing the test.']);
        param.useTestData = false;
    end
end

%% the main loop
while 1
    % properties of dictionary now
    it = it + 1;
    propD = dictprop(D, false);   % without gap-properties
    res.fbA(it) = propD.A;
    res.fbB(it) = propD.B;
    res.betamin(it) = propD.betamin;
    res.betaavg(it) = propD.betaavg;
    res.betamse(it) = propD.betamse;
    if isfield(param,'min_diagC'); res.min_diagC(it) = min(diag(C)); end;
    if isfield(param,'max_diagC'); res.max_diagC(it) = max(diag(C)); end;
    %
    if (it > param.mainIt) 
        break; 
    end
    %
    X = getXfromBSDS300(parX);    % get new data
    if (param.verbose > 0)
        fprintf('Main iteration %i : new X is %3i x %5i \n',...
                 it, size(X,1), size(X,2) );
    end
    %
    if isfield(param,'avg_w0'); sum_w0 = 0; end;
    if isfield(param,'avg_w1'); sum_w1 = 0; end;
    if isfield(param,'avg_w2'); sum_w2 = 0; end;
    if isfield(param,'avg_r0'); sum_r0 = 0; end;
    if isfield(param,'avg_r1'); sum_r1 = 0; end;
    if isfield(param,'avg_r2'); sum_r2 = 0; end;
    if isfield(param,'avg_rr'); sum_rr = 0; end;
    if (param.useTestData == false)
        if isfield(param,'avg_x0'); res.avg_x0(it) = mean(sum(X ~= 0)); end;
        if isfield(param,'avg_x1'); res.avg_x1(it) = mean(sum(abs(X))); end;
        if isfield(param,'avg_x2'); res.avg_x2(it) = mean(sqrt(sum(X.*X))); end;
    end
    %
    if strcmpi(param.method, 'javaRLS')
        jDL.rlsdla( X(:), int32(1) );   % process X once
        D = reshape(jDL.getDictionary.getAll(), N, K);
        jD.setAll(D(:));
        jDD.eqInnerProductMatrix(jD);
        jMP.checkNormalized();
    else   % RLS in matlab
        j = 0;  % the vector number to process
        bsit =  param.dataBatch( min(it,numel(param.dataBatch)) );  % i.e. batchsize for this iteration
        % use lam0b = lambda0^bsit approx (1-bsit*(1-lambda0))  (0 <= bsit*(1-lambda0) < 0.5)
        if ((bsit*(1-param.lambda0)) < 0.5)
            lam0b = 1-bsit*(1-param.lambda0);
        else
            lam0b = param.lambda0^bsit;
        end
        for bno = 1:floor(size(X,2)/bsit)  % size(X,2) = param.dataLength
            lambda = lambdafun(j+(it-1)*param.dataLength, 'C', 0.9*param.vectorsToProcess, lam0b, 1);
            if (param.verbose > 1) && (bno < 4)
                fprintf('it=%i bno=%i : lambda used once for each %i vectors is now %8.6f \n',...
                    it, bno, bsit, lambda );
            end
            C = (1/lambda)*C;
            if strcmpi(param.vecSel(1:3), 'mex')
                if strcmpi(param.vecSel, 'mexOMP')
                    W = mexOMP(X(:,j+(1:bsit)), D, param.paramOMP);
                elseif strcmpi(param.vecSel, 'mexLasso')
                    W = mexLasso(X(:,j+(1:bsit)), D, param.paramLasso);
                else
                    error([mfile,': param.vecSel = ',param.vecSel,' (illegal).']);
                end
                jw = 0;
                for k=1:bsit    % for each element in this batch
                    j = j+1;
                    jw = jw+1;
                    if (j > size(X,2)); break; end;
                    x = X(:,j);
                    w = W(:,jw);
                    r = x - D*w;
                    rr = r'*r;
                    if (param.useTestData == false)
                        if isfield(param,'avg_w0'); sum_w0 = sum_w0 + sum(w ~= 0); end;
                        if isfield(param,'avg_w1'); sum_w1 = sum_w1 + sum(abs(w)); end;
                        if isfield(param,'avg_w2'); sum_w2 = sum_w2 + sqrt(w'*w); end;
                        if isfield(param,'avg_r0'); sum_r0 = sum_r0 + sum(r ~= 0); end;
                        if isfield(param,'avg_r1'); sum_r1 = sum_r1 + sum(abs(r)); end;
                        if isfield(param,'avg_r2'); sum_r2 = sum_r2 + sqrt(rr); end;
                        if isfield(param,'avg_rr'); sum_rr = sum_rr + rr; end;
                    end
                    u = C*w;
                    alpha = 1/(1+w'*u);
                    alphaut = alpha*u';
                    D = D + r*alphaut;
                    C = C - u*alphaut;
                end
            else % java
                for k=1:bsit   % for each element in this batch
                    j = j+1;
                    if (j > size(X,2)); break; end;
                    x = X(:,j);
                    normx = sqrt(x'*x);
                    if (normx < param.absErrorLimit); continue; end;
                    if strcmpi(param.vecSel, 'javaORMP')
                        w = jMP.vsORMP(x, param.sparseNum, param.absErrorLimit/normx);
                    elseif strcmpi(param.vecSel, 'javaOMP')
                        w = jMP.vsOMP(x, param.sparseNum, param.absErrorLimit/normx);
                    end
                    r = x - D*w;
                    rr = r'*r;
                    if (param.useTestData == false)
                        if isfield(param,'avg_w0'); sum_w0 = sum_w0 + sum(w ~= 0); end;
                        if isfield(param,'avg_w1'); sum_w1 = sum_w1 + sum(abs(w)); end;
                        if isfield(param,'avg_w2'); sum_w2 = sum_w2 + sqrt(w'*w); end;
                        if isfield(param,'avg_r0'); sum_r0 = sum_r0 + sum(r ~= 0); end;
                        if isfield(param,'avg_r1'); sum_r1 = sum_r1 + sum(abs(r)); end;
                        if isfield(param,'avg_r2'); sum_r2 = sum_r2 + sqrt(rr); end;
                        if isfield(param,'avg_rr'); sum_rr = sum_rr + rr; end;
                    end
                    u = C*w;
                    alpha = 1/(1+w'*u);
                    alphaut = alpha*u';
                    D = D + r*alphaut;
                    C = C - u*alphaut;
                end
            end
            % normalize (rescale) the dictionary
            diagG = 1./sqrt(sum(D.^2));
            D = D .* repmat(diagG, N, 1);
            % rescale C and make sure it is symmetric
            for k1 = 1:K
                for k2 = k1:K
                    temp = C(k1,k2)*(diagG(k1)*diagG(k2));
                    C(k1,k2) = temp;
                    C(k2,k1) = temp;
                end
            end
            if strcmpi(param.vecSel(1:4), 'java')
                % update java matrices used for vector selection
                jD.setAll(D(:));
                jDD.eqInnerProductMatrix(jD);
            end
        end        %  end of  for bno ...
        %
        if (param.verbose == 1)
            fprintf('it=%i bno=%i : lambda used once for each %i vectors is now %8.6f \n',...
                it, bno, bsit, lambda );
        end
        %
        % -- special ad-hoc adjustment of D -- START
        if sum(it == [1,2,5,10,15,21,41,91,191,401,801]) > 0
            % only change order of columns in D
            [temp,p] = sort(diag(C), 'ascend');  % sort smallest first
            % p defines a permutation matrix: P(i, p(i)) = 1
            C = C(p,p);     % P*C*P'
            D = D(:,p);     % D*P', most used vectors first in D
            if strcmpi(param.vecSel(1:4), 'java')
                jD.setAll(D(:));
                jDD.eqInnerProductMatrix(jD);
            end
        end
        % -- special ad-hoc adjustment of D -- END
    end
    %
    % result after each set of training data is processed
    if param.useTestData
        if (sum(isfield(param,{'avg_w0','avg_w1','avg_w2','avg_r0','avg_r1','avg_r2'})) > 0)
            if strcmpi(param.vecSel(1:4), 'java')
                for j = 1:size(Xtest,2);    % for each element
                    x = Xtest(:,j);
                    if sqrt(x'*x) < param.absErrorLimit
                        w = zeros(K,1);
                    else    % if strcmpi(param.vecSel, 'javaORMP')
                        % use javaORMP in test also for javaOMP in learning 
                        w = jMP.vsORMP(x, param.sparseNum, param.absErrorLimit/sqrt(x'*x));
                    % elseif strcmpi(param.vecSel, 'javaOMP')
                    %     w = jMP.vsOMP(x, param.sparseNum, param.absErrorLimit/sqrt(x'*x));
                    end
                    r = x - D*w;  
                    rr = r'*r;
                    if isfield(param,'avg_w0'); sum_w0 = sum_w0 + sum(w ~= 0); end;
                    if isfield(param,'avg_w1'); sum_w1 = sum_w1 + sum(abs(w)); end;
                    if isfield(param,'avg_w2'); sum_w2 = sum_w2 + sqrt(w'*w); end;
                    if isfield(param,'avg_r0'); sum_r0 = sum_r0 + sum(r ~= 0); end;
                    if isfield(param,'avg_r1'); sum_r1 = sum_r1 + sum(abs(r)); end;
                    if isfield(param,'avg_r2'); sum_r2 = sum_r2 + sqrt(rr); end;
                    if isfield(param,'avg_rr'); sum_rr = sum_rr + rr; end;
                end
            else  % mex
                W = mexOMP(Xtest, D, param.paramOMP);   % use mexOMP in test also for mexLasso in learning
                R = Xtest - D*W;
                if isfield(param,'avg_w0'); res.avg_w0(it) = mean(sum(W ~= 0)); end;
                if isfield(param,'avg_w1'); res.avg_w1(it) = mean(sum(abs(W))); end;
                if isfield(param,'avg_w2'); res.avg_w2(it) = mean(sqrt(sum(W.*W))); end;
                if isfield(param,'avg_r0'); res.avg_r0(it) = mean(sum(R ~= 0)); end;
                if isfield(param,'avg_r1'); res.avg_r1(it) = mean(sum(abs(R))); end;
                if isfield(param,'avg_r2'); res.avg_r2(it) = mean(sqrt(sum(R.*R))); end;
                if isfield(param,'avg_rr'); res.avg_rr(it) = mean(sum(R.*R)); end;
                clear W R
            end
        end
    end
    if strcmpi(param.vecSel(1:4), 'java')
        if isfield(param,'avg_w0'); res.avg_w0(it) = sum_w0/j; end;
        if isfield(param,'avg_w1'); res.avg_w1(it) = sum_w1/j; end;
        if isfield(param,'avg_w2'); res.avg_w2(it) = sum_w2/j; end;
        if isfield(param,'avg_r0'); res.avg_r0(it) = sum_r0/j; end;
        if isfield(param,'avg_r1'); res.avg_r1(it) = sum_r1/j; end;
        if isfield(param,'avg_r2'); res.avg_r2(it) = sum_r2/j; end;
        if isfield(param,'avg_rr'); res.avg_rr(it) = sum_rr/j; end;
    end
    %
    res.vectorsProcessed = res.vectorsProcessed + size(X, 2);
    % the Frobenius norm does not work well when ad-hoc update of D is done
    % res.delta0(it) = norm( D - D0, 'fro');
    % res.deltai(it) = norm( D - res.D, 'fro');
    res.delta0(it) = dictdiff(D, D0, 'mean', 'thabs')*180/pi;
    res.deltai(it) = dictdiff(D, res.D, 'mean', 'thabs')*180/pi;
    res.D = D;
    if isfield(param,'avg_rr'); 
        disp(['--   iteration ',int2str(it),' done ',datestr(now()),...
            '  delta-Di = ',sprintf('%5.2f', res.deltai(it)), ...
            '  w0-norm = ',sprintf('%5.2f', res.avg_w0(it)), ...
            '  PSNR = ',sprintf('%5.2f', 10*log10((255^2)/(res.avg_rr(it)/N)))]);
    else
        disp(['--   iteration ',int2str(it),' done ',datestr(now())]);
    end
    if exist('stopp.fil','file')  % a controlled stop of the iterations
        res.aborted = true;
        break; 
    end
end
res.endTime = datestr(now());
res.totalTime = toc(tstart);

if (isfield(param, 'saveResInFile') && param.saveResInFile)
    dstr = datestr(now());
    resultFile = [mfile,'_',dstr([4:6,1,2,13,14,16,17]),'.mat'];
    res.resultFile = resultFile;
    if (param.verbose >= 0); disp(['Save results in ',resultFile]); end;
    save(resultFile, 'res');
end

return


%% ****** notes on MSE and PSNR
% PSNR (Peak Signal to Noise Ratio) is (in Matlab code, given X, D and W)
% PSNR = 10*log10( (numel(X)*param.dataPeak^2) / sum(sum( (X-D*W).^2 )) );
% MSE is E{r'r}, where E{.} is expectation, and r = x - Dw
% E{r'r} is actually (in Latex) $ 1/L \sum_{i=1}^L r_i^T r_i $ 
% or in Matlab: E{r'r} = sum(sum( (X-D*W).^2 ))/size(X,2);
%  SSE = sum(sum( (X-D*W).^2 ));  % sum squared error
%  MSE = SSE/numel(X);
%  SSp = numel(X)*param.dataPeak^2;  % sum squared peak
%  PSNR = 10*log10(SSp/SSE);  % = 10*log10( param.dataPeak^2/MSE )

%% om sortering
% W = randn(7,20);
% D = [1:7; 8:14];
% C = W*W';
% [temp,p] = sort(diag(C));  % sort smallest first
% p = flipud(p);  % reverse is largest first
% P = makeP(p);   % P*diag(C)  sort (rows) largest first
% Wp = P*W;
% Cp = P*C*P';    
% Dp = D*P';      % same order for D (as for C)
% x = D*W;
% xp = Dp*Wp;
% % without P matrix
% Cp2 = C(p,p);
% Dp2 = D(:,p);
% Wp2 = W(p,:);
% xp2 = Dp2*Wp2;
% % all these should be zero
% disp(norm(Dp2-Dp));
% disp(norm(Cp2-Cp));
% disp(norm(Wp2-Wp));
% disp(norm(xp2-xp));
% disp(norm(xp2-x));


