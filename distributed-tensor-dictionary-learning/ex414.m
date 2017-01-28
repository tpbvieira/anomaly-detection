% ex414           Train dictionaries for BSDS300 images, jDL.rlsdla(..)
%                 Note that this is a command file, not a function, so it 
% uses the variables in workspace. The important variables, which are set
% to default values if they do not exist, are:
%  dataSet     : A (D is 64x256), B (D is 432x512) or C (D is 256x1024)
%                'lena' is a special variant
%  batches     : number of batches to do, a batch here is not the same as
%                a minibatch used in ODL and LS-DLA.
%  dataLength  : number of TV (training vectors) to process in each batch
%                default is 10000 (which is usually appropriate)
%  lambda0     : initial value for lambda, default is 0.9995 
%  lambdaa     : number of TV to process before lambda is 1
%                default is 0.95*batches*dataLength 
%  vecSel      : 'javaORMP' (default) or 'javaOMP' 
%  targetPSNR  : default 38 (is ok for dataSet A), 
%                should be smaller, for example 28,  for dataSet B and C
%  D           : initial dictionary, default (from data) is usually good.
%                Initial C matrix is not given.
% 
% To stop execution in a controlled way, create a file: 'stopp.fil'
% The results will be stored in 'ex414_mmmddhhmm.mat'.
%
% examples:  (estimated times for 1 Mtv: hh:mm are A: 00:15, B: hh:mm,  C: hh:mm)
% clear all; ex414;  % all defaults
% clear all; batches=250; dataLength=20000; lambda0=0.996; targetPSNR=40; ex414;  

% clear all; dataSet='lena'; dataLength=2000; batches=100; lambda0=0.998; targetPSNR=38; ex414;   
% clear all; dataSet='B'; batches=500; lambda0=0.9990; targetPSNR=34; ex414;  
% clear all; dataSet='C'; dataLength=5120; batches=1000; lambda0=0.9995; targetPSNR=31; ex414;   
%----------------------------------------------------------------------
% Karl Skretting. 
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  07.01.2011  KS: test started
% Ver. 1.1  26.04.2011  KS: rewritten test (to minimize storage)
% Ver. 1.2  02.09.2011  KS: minor changes to make it simpler
%----------------------------------------------------------------------

mfile = 'ex414';
if exist('stopp.fil','file')  
    disp(['Delete ''stopp.fil'' to run ',mfile]);
    return;
end
if ~exist('dataSet', 'var'); dataSet = 'A'; end;
if ~exist('batches', 'var'); batches = 50;   end;      %  number of batches 
if ~exist('dataLength', 'var'); dataLength = 10000; end;  % number of TV to prosess in each batch
if ~exist('lambda0', 'var'); lambda0 = 0.996; end;
if ~exist('lambdaa', 'var'); lambdaa = 0.95*dataLength*batches; end;
if ~exist('vecSel', 'var'); vecSel = 'javaORMP';  end;   % javaORMP or javaOMP
if ~exist('targetPSNR', 'var'); targetPSNR = 38;  end;   
if ~exist('transform', 'var'); transform = 'none';  end;   

if       strcmpi(dataSet,'A');   N = 8*8;      K = 256;  parXn = [8,8];    col = false; 
elseif   strcmpi(dataSet,'B');   N = 12*12*3;  K = 512;  parXn = [12,12];  col = true;  
elseif   strcmpi(dataSet,'C');   N = 16*16;    K = 1024; parXn = [16,16];  col = false; 
elseif   strcmpi(dataSet,'lena');   N = 8*8;   K = 256;  parXn = [8,8];    col = false; 
end
targetMSE = N * 255^2 * 10^(-targetPSNR/10);
sparseNum = int32(floor(min(0.5*N,90))); % ORMP returns when ||w||_0 == param.sparseNum
absErrorLimit = sqrt(targetMSE);    % ORMP returns when ||r||_2 <= param.absErrorLimit

% the parameters used in getXfromBSDS300
parX = struct('imCat','train', 'no', dataLength, ...
              'myim2colPar',struct('n',parXn, 'i',parXn, 'a','none', 'v',0), ...
              'transform',transform, 'col',col, 'sm',true, 'norm1lim', 5);

java_access;

if exist('D', 'var') && (size(D,1) == N)  && (size(D,2) == K)
    jD = mpv2.SimpleMatrix( D );
elseif strcmpi(dataSet,'lena')
    % X = getXfrom8images('images', {'lena.bmp'}, 'noFromEachIm',K);
    % X = X - repmat(mean(X), size(X,1), 1);   % subtract mean
    X = get64x256() + 0.1*randn(64,256);
    jD = mpv2.SimpleMatrix( X );
else
    jD = mpv2.SimpleMatrix( getXfromBSDS300(parX, 'no',K) );
end
jDL  = mpv2.DictionaryLearning(jD, true);  % store (D'*D)
jDL.setLambda('Q', lambda0, 1.0, lambdaa);
if strcmpi(vecSel,'javaORMP') || strcmpi(vecSel,'ORMP')
    jDL.setORMP(sparseNum, 1e-6, absErrorLimit);
else
    jDL.setOMP(sparseNum, 1e-6, absErrorLimit);
end
jDL.setLoggOff();
jDL.setVerbose(0);
jDL.setParamI1(16);

res = struct(  'startTime', datestr(now()), ...
               'endTime', datestr(now()), ...
               'totalTime', -1, ...
               'inputParam', struct('mfile',mfile, 'vecSel',vecSel, 'dataSet',dataSet, ...
                          'lambda0',lambda0, 'lambdaa',lambdaa, 'targetPSNR',targetPSNR, ...
                          'transform',transform, 'batches',batches, 'dataLength',dataLength ), ...
               'vectorsProcessed', 0, ...
               'computer', computer(), ...
               'parX', parX, ...
               'fbA', zeros(1, batches+1), ...  
               'fbB', zeros(1, batches+1), ...
               'mu', zeros(1, batches+1), ...
               'muavg', zeros(1, batches+1), ...
               'mumse', zeros(1, batches+1), ...
               'deltai', zeros(1, batches), ...  % ||D_i - D_{i-1}||
               'D', zeros(N,K) );                     % final dictioary
%           
res.D = reshape(jDL.getDictionary.getAll(), N, K);
bno = 0;
propD = dictprop(res.D, false);   % without gap-properties
res.fbA(bno+1) = propD.A;
res.fbB(bno+1) = propD.B;
res.mu(bno+1) = propD.mu;
res.muavg(bno+1) = propD.muavg;
res.mumse(bno+1) = propD.mumse;

tv0 = jDL.getNoTV();
tic;
for bno = 1:batches
    if strcmpi(dataSet,'lena')
        X = getXfrom8images('images', {'lena.bmp'}, 'noFromEachIm',dataLength);
        X = X - repmat(mean(X), size(X,1), 1);   % subtract mean
    else
        X = getXfromBSDS300(parX);
    end
    %
    jDL.rlsdla( X(:), int32(1) );   % process X once
    %
    % several properties may be reported on the way
    res.deltai(bno) = dictdiff(res.D, reshape(jDL.getDictionary.getAll(), N, K), 'mean', 'thabs')*180/pi;
    %
    res.D = reshape(jDL.getDictionary.getAll(), N, K);
    propD = dictprop(res.D, false);   % without gap-properties
    res.fbA(bno+1) = propD.A;
    res.fbB(bno+1) = propD.B;
    res.mu(bno+1) = propD.mu;
    res.muavg(bno+1) = propD.muavg;
    res.mumse(bno+1) = propD.mumse;
    %
    disp(['Batch number = ',int2str(bno), ...
          ', ',datestr(now()), ...
          ', deltai=',sprintf('%6.3f', res.deltai(bno)), ...
          ', B=',sprintf('%6.2f',propD.B), ...
          ', mu=',sprintf('%6.4f',propD.mu), ...
          ', muavg=',sprintf('%6.4f',propD.muavg), ...
          ', mumse=',sprintf('%6.4f',propD.mumse) ]);
    %
    if exist('stopp.fil','file')  % a controlled stop of the iterations
        res.deltai = res.deltai(1:bno);
        res.fbA = res.fbA(1:(bno+1));
        res.fbB = res.fbB(1:(bno+1));
        res.mu = res.mu(1:(bno+1));
        res.muavg = res.muavg(1:(bno+1));
        res.mumse = res.mumse(1:(bno+1));
        break; 
    end
end
res.totalTime = toc;
tv1 = jDL.getNoTV();
disp(['dataSet ',dataSet,': TVs per second = ',num2str((tv1-tv0)/res.totalTime)]);
t=testDLplot(res, 0, 0);
res.SRC = t(9);
res.PSNR = t(10);

res.vectorsProcessed = tv1-tv0;
res.endTime = datestr(now());
%
% if ~exist('resultFile', 'var')
dstr = datestr(now());
resultFile = [mfile,'_',dstr([4:6,1,2,13,14,16,17]),'.mat'];
res.resultFile = resultFile;
save(resultFile, 'res');

return

