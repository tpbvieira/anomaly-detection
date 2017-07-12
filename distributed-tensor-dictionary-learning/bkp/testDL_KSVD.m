function res = testDL_KSVD(param)
% testDL_KSVD     Test K-SVD Dictionary Learning, Berkeley Segmentation Dataset
%
% Vector selection method is given by: param.vecSel
% which may be: 'javaOMP', 'javaORMP', 'mexOMP', 'mexLasso'
% ('mexOMP' is the same as ORMP, and gives same results as 'javaORMP')
%
% The Berkeley Segmentation Dataset is used, 
% the 3 possible training datasets (A, B and C) are thus (almost) infinite,
% but K-SVD uses same data as MOD from a fixed set of 10^6 training vectors. 
% Assume 50 datafiles are available, BUT use maximum 10^5 in each iteration.
% The results are reported after each main iteration, i.e.
% after eache set of training data is processed.
%
% The results are also saved in 'testDL_KSVD_mmmddhhmm.mat'.
% Result can be displayed by: testDLplot(res)
% 
% res = testDL_KSVD(param);
%----------------------------------------------------------------------
% The input argument param is simply a struct with appropriate fields
%   actual PSNR will often be 1 to 1.5 dB above the targetPSNR
%   avg_?? tells which properties to log during learning for
%          test data (if useTestData) or for training data
%          ex: ?? = w0 for average ||w||_0 (for the test/training vectors)
% The output argument is also a struct with results (for each iteration)
%----------------------------------------------------------------------
% examples: 
% res = testDL_KSVD();   % all default
% param = struct('vecSel','mexOMP', 'mainIt',500, 'targetPSNR', 38, ...
%                'useTestData',true, 'verbose', 0, ...
%                'avg_w0',1, 'avg_w1',1, 'avg_r2',1, 'avg_rr',1 );
% res = testDL_KSVD( param );  
% testDLplot( res );

%----------------------------------------------------------------------
% Karl Skretting. 
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  10.03.2011  KS: made function based on testDL_MOD
% Ver. 1.1  07.09.2011  KS: complete path to dataKatalog should be given
%----------------------------------------------------------------------

mfile = 'testDL_KSVD';
if isunix()
    catSep = '/';
    dataKatalog = 'D:/Karl/DICT/matlab/d2';
else
    catSep = '\';
    dataKatalog = 'datacat';   
end

% fields with fixed values
param.mfile = mfile;
param.method = 'K-SVD';
param.dataPeak = 255;
param.saveResInFile = true;
param.dataLength = 20000;   % length of each dataset 
% fields with default values which may be given in input
if ~isfield(param,'vecSel');       param.vecSel        = 'javaORMP'; end;
if ~isfield(param,'mainIt');       param.mainIt        = 20;         end;
if ~isfield(param,'dataSet');      param.dataSet       = 'A';        end;
if ~isfield(param,'targetPSNR');   param.targetPSNR    = 38;         end;
if ~isfield(param,'verbose');      param.verbose       = 0;          end;
if ~isfield(param,'useTestData');  param.useTestData   = false;      end;
if param.mainIt <= 20
    param.dataBatch = ones(1,param.mainIt);
else
    param.dataBatch = ones(1,param.mainIt);
    param.dataBatch ( floor(0.45*param.mainIt):end ) = 2;
    param.dataBatch ( floor(0.85*param.mainIt):end ) = 3;
    param.dataBatch ( floor(0.95*param.mainIt):end ) = 4;
end

% % if 'useTestData' is true results in 'avg_*' will be from test data, 
% % else these results will be from training data
% fields which normally is not used but may be given in input:
%  seedD0, avg_w0, avg_w1, avg_w2, avg_x0, avg_x1, avg_x2, 
%  avg_r0, avg_r1, avg_r2, min_diagC, max_diagC
% fields/variables derived from the input parameters (param)
if       (param.dataSet == 'A');   N = 8*8;      K = 256;  n = [8,8];    col = false;
elseif   (param.dataSet == 'B');   N = 12*12*3;  K = 512;  n = [12,12];  col = true;
elseif   (param.dataSet == 'C');   N = 16*16;    K = 1024; n = [16,16];  col = false; 
end
param.targetMSE = N * param.dataPeak^2 * 10^(-param.targetPSNR/10);
param.sparseNum = int32(floor(0.5*N));   % ORMP returns when ||w||_0 == param.sparseNum
param.absErrorLimit = sqrt(param.targetMSE);    % ORMP returns when ||r||_2 <= param.absErrorLimit

if strcmpi(param.vecSel(1:4), 'java')
    % check access to java
    java_access();
end
if strcmpi(param.vecSel(1:3), 'mex')
    if strcmpi(param.vecSel, 'mexORMP'); param.vecSel = 'mexOMP'; end;
    if strcmpi(param.vecSel, 'mexLARS'); param.vecSel = 'mexLasso'; end;
    % check access to SPAMSS 
    if isunix()
        SPAMScatalog = '~/Matlab/SPAMS/release/mkl64/';
    else
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

parX = struct('imCat','train', 'no', param.dataLength, ...
              'myim2colPar',struct('n',n, 'i',[5,5], 'a','none', 'v',0), ...
              'col',col, 'sm',true, 'norm1lim', 5);
% make files for learning data if they not already exist
datafile = [dataKatalog,catSep,'data',param.dataSet,'01.mat'];
if ~exist(datafile,'file')
    disp([mfile,': can not locate large set of training vectors (',...
        datafile,'), and do not create it here.']);
    disp('Random sets will be generated in each iteration using getXfromBSDS300(parX).'); 
    % this is how a large set of training vectors may be generated
    % for i=1:50 
    %     datafile = [dataKatalog,catSep,'data',param.dataSet,sprintf('%02i',i),'.mat'];
    %     X = getXfromBSDS300(parX);  
    %     save(datafile,'X');
    % end
end

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

disp(' ');
disp(['---- ',mfile,' started ',res.startTime]);

%% initial Dictionary, D0, D and res.D, and C 
tstart = tic;
if isfield(param,'seedD0');          
    RandStream.setDefaultStream(RandStream('mt19937ar','seed',param.seedD0));
end
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
res.D = D0;

if strcmpi(param.vecSel(1:4), 'java')
    % initialize dictionary learning for java
    jD = mpv2.SimpleMatrix(D0);
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
            Xtest = Xtest.X;
            if isfield(param,'avg_x0'); res.avg_x0 = mean(sum(Xtest ~= 0)); end;
            if isfield(param,'avg_x1'); res.avg_x1 = mean(sum(abs(Xtest))); end;
            if isfield(param,'avg_x2'); res.avg_x2 = mean(sqrt(sum(Xtest.*Xtest))); end;
        else
            disp([mfile,': ',testDataFile,' does not contain correct test data.']);
            clear Xtest
            param.useTestData = false;
        end
    else
        disp([mfile,': does not find ',testDataFile,'.']);
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
    %
    if (it > param.mainIt) 
        break; 
    end
    %
    % Convergence improved, but not results on test data if test below is
    % used. Thus the test is removed now (KS: March 11)
    % if (it < 0.95*param.mainIt)        % use the same X for the last iterations
        useSett = randperm( 50 );
        useSett = useSett( 1:param.dataBatch(min(it,numel(param.dataBatch))) );
        X = zeros(N,numel(useSett)*param.dataLength);
        if ~exist([dataKatalog,catSep,'data',param.dataSet,'01.mat'],'file')
            i1 = 0;
            for i = useSett
                temp = getXfromBSDS300(parX);
                X(:,i1+(1:size(temp,2))) = temp;
                i1 = i1 + size(temp,2);
            end
            clear temp
            if size(X,2) ~= i1
                disp('warning: X was not expected size.');
                X = X(:,1:i1);
            end
            disp(['---  iteration ',sprintf('%4i',it),' uses X of size ', ...
                int2str(size(X,1)),'x',int2str(size(X,2)), ...
                ' with random patches from training images.']);
        else   % since first exist we assume all 50 exist
            i1 = 0;
            for i = useSett
                datafile = [dataKatalog,catSep,'data',param.dataSet,sprintf('%02i',i),'.mat'];
                temp = load(datafile);
                X(:,i1+(1:size(temp.X,2))) = temp.X;
                i1 = i1 + size(temp.X,2);
            end
            clear temp
            if size(X,2) ~= i1
                disp('warning: X was not expected size.');
                X = X(:,1:i1);
            end
            % end
            temp = sprintf('%02i, ',useSett);
            disp(['---  iteration ',sprintf('%4i',it),' uses X of size ', ...
                int2str(size(X,1)),'x',int2str(size(X,2)), ...
                ' with data from files: ',temp(1:(end-2))]);
        end
        %
    if strcmpi(param.vecSel(1:3), 'mex')
        if strcmpi(param.vecSel, 'mexOMP')
            W = mexOMP(X, D, param.paramOMP);
        elseif strcmpi(param.vecSel, 'mexLasso')
            W = mexLasso(X, D, param.paramLasso);
        else
            error([mfile,': param.vecSel = ',param.vecSel,' (illegal).']);
        end
    end
    %
    if strcmpi(param.vecSel(1:4), 'java')
        W = zeros(K,size(X,2));
        for j = 1:size(X,2)
            x = X(:,j);
            normx = sqrt(x'*x);
            if (normx < param.absErrorLimit)
                w = zeros(K,1);
            elseif strcmpi(param.vecSel, 'javaORMP')
                w = jMP.vsORMP(x, param.sparseNum, param.absErrorLimit/normx);
            elseif strcmpi(param.vecSel, 'javaOMP')
                w = jMP.vsOMP(x, param.sparseNum, param.absErrorLimit/normx);
            end
            W(:,j) = w;
        end
    end
    %
    W = full(W);
    if strcmpi(param.vecSel, 'mexLasso') || isfield(param,'useMOD') 
        % MOD must be used for mexLasso (since K-SVD the destroy ||w||_1 minimum) 
        disp('--  Note MOD instead of K-SVD in this case.');
        R = X - D*W;
        D = (X*W')/(W*W');   % MOD equation
        diagG = 1./sqrt(sum(D.^2));  
        D = D .* repmat(diagG, N, 1); % and normalization
    else
        % Dictionary (and W) update using the K-SVD method
        % R = X - D*W;
        % disp(['K-SVD, SSE before K-SVD : ',sprintf('%5g',sum(sum(R.*R)))]);
        for k=1:K
            R = X - D*W;
            Ik = find(W(k,:));
            Rk = R(:,Ik) + D(:,k)*W(k,Ik);
            [U,S,V] = svds(Rk,1,'L');
            D(:,k) = U;
            W(k,Ik) = S*V';
        end
        R = X - D*W;
        % disp(['K-SVD, SSE after  K-SVD : ',sprintf('%5g',sum(sum(R.*R)))]);
    end
    
    if (param.useTestData == false)
        if isfield(param,'avg_x0'); res.avg_x0(it) = nnz(X)/size(X,2); end;
        if isfield(param,'avg_x1'); res.avg_x1(it) = sum(abs(X(:)))/size(X,2); end;
        if isfield(param,'avg_x2'); res.avg_x2(it) = sum(sqrt(sum(X.*X)))/size(X,2); end;
        if isfield(param,'avg_w0'); res.avg_w0(it) = nnz(W)/size(X,2); end;
        if isfield(param,'avg_w1'); res.avg_w1(it) = sum(abs(W(:)))/size(X,2); end;
        if isfield(param,'avg_w2'); res.avg_w2(it) = sum(sqrt(sum(W.*W)))/size(X,2); end;
        if isfield(param,'avg_r0'); res.avg_r0(it) = nnz(R)/size(X,2); end;
        if isfield(param,'avg_r1'); res.avg_r1(it) = sum(abs(R(:)))/size(X,2); end;
        if isfield(param,'avg_r2'); res.avg_r2(it) = sum(sqrt(sum(R.*R)))/size(X,2); end;
        if isfield(param,'avg_rr'); res.avg_rr(it) = sum(sum(R.*R))/size(X,2); end;
    end
    
    if strcmpi(param.vecSel(1:4), 'java')
        % update java matrices used for vector selection
        jD.setAll(D(:));
        jDD.eqInnerProductMatrix(jD);
    end     
    %
    % result after each set of training data is processed
    if param.useTestData
        if (sum(isfield(param,{'avg_w0','avg_w1','avg_w2','avg_r0','avg_r1','avg_r2'})) > 0)
            if strcmpi(param.vecSel(1:4), 'java')
                W = zeros(K, size(Xtest,2));
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
                    W(:,j) = w;
                end
            else  % mex
                W = mexOMP(Xtest, D, param.paramOMP);   % use mexOMP in test also for mexLasso in learning
            end
            R = Xtest - full(D*W);
            if isfield(param,'avg_w0'); res.avg_w0(it) = mean(sum(W ~= 0)); end;
            if isfield(param,'avg_w1'); res.avg_w1(it) = mean(sum(abs(W))); end;
            if isfield(param,'avg_w2'); res.avg_w2(it) = mean(sqrt(sum(W.*W))); end;
            if isfield(param,'avg_r0'); res.avg_r0(it) = mean(sum(R ~= 0)); end;
            if isfield(param,'avg_r1'); res.avg_r1(it) = mean(sum(abs(R))); end;
            if isfield(param,'avg_r2'); res.avg_r2(it) = mean(sqrt(sum(R.*R))); end;
            if isfield(param,'avg_rr'); res.avg_rr(it) = mean(sum(R.*R)); end;
        end
    end
    %
    res.vectorsProcessed = res.vectorsProcessed + size(X, 2);
    res.delta0(it) = dictdiff(D, D0, 'mean', 'thabs')*180/pi;
    res.deltai(it) = dictdiff(D, res.D, 'mean', 'thabs')*180/pi;
    res.D = D;
    if isfield(param,'avg_rr'); 
        disp(['--   iteration ',sprintf('%4i',it),' done ',datestr(now()),...
            '  delta-Di = ',sprintf('%5.2f', res.deltai(it)), ...
            '  w0-norm = ',sprintf('%5.2f', res.avg_w0(it)), ...
            '  PSNR = ',sprintf('%5.2f', 10*log10((255^2)/(res.avg_rr(it)/N)))]);
    else
        disp(['--   iteration ',sprintf('%4i',it),' done ',datestr(now())]);
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


