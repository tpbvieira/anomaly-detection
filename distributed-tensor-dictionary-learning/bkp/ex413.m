function res = ex413(varargin)
% ex413           Train dictionaries from images, ODL/(R)LS and minibatches
%
% To stop execution in a controlled way, create a file: 'stopp.fil'
% The results will be stored in 'ex413_mmmddhhmm.mat'.
% 
% res = ex413(argName, argVal, ...);
%----------------------------------------------------------------------
%        The input arguments (options) may be:
%  dataSet     : A (D is 64x256), B (D is 432x512) or C (D is 256x1024)
%  minibatch   : number of minibatches to do, each row is [nof batches, batchsize]
%                default: [200,50; 500,100; 6200,200; 1400,500]; 
%  parX        : a struct with parameters in getXfromBSDS300.m
%  vecSel      : method to use for vector selection: 'mexOMP', 'mexLasso', 'javaORMP'
%  targetPSNR  : default 38 
%  ODLvariant  : 1 - ODL as Mairal-paper, only normalization of large atoms
%                2 - ODL, normalization of all atoms (not A and B matrices)
%                3 - does not work quite correctly ???
%                4 - does not work quite correctly !!!
%                5 - (R)LS-DLA minibatch, without normalization of A and B
%                6 - (R)LS-DLA minibatch, normalization of A and B
%              All variants are with Search-Then-Converge scheme: 
%  lambda0     : initial value for lambda, default is 0.998 
%  lambdaa     : number of TV to process before lambda is 1
%                default is 0.95*sum(minibatch(:,1).*minibatch(:,2))
%  D           : initial dictionary from data is usually good.
%                to put some weight at this in the beginning it should be
%                scaled up to match the training vectors.
%                ex:  load dict_ahoc; % D is loaded to workspace
%                     res = ex413(... 'D',1000*D);
%----------------------------------------------------------------------
%
% examples:  (estimated times for 1 Mtv: hh:mm are A: hh:mm, B: hh:mm,  C: hh:mm)
% r = ex413();     % all defaults
% % total number of vectors processed is: sum(mb(:,1).*mb(:,2))
% mb = [2000,50; 5000,100; 10000,200; 5000,250; 2300,500];
% p5 = struct('vecSel','mexOMP', 'lambda0',0.996, 'targetPSNR',40, ...
%             'ODLvariant',5, 'minibatch',mb); 
% res5 = ex413(p5);
% p6 = p5; p6.ODLvariant = 2;
% res6 = ex413(p6);
% p7 = struct('vecSel','mexLasso', 'lambda0',0.998, 'targetPSNR',35, ...
%             'ODLvariant',5, 'minibatch',mb); 
% res7 = ex413(p7);
% p8 = p7; p8.ODLvariant = 2;
% res8 = ex413(p8);

%----------------------------------------------------------------------
% Karl Skretting. 
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  13.05.2011  KS: test started
% Ver. 1.1  08.09.2011  KS: Works ok
%----------------------------------------------------------------------
%  Matrices:
%  A           : sum( w*w' ) 
%  B           : sum( x*w' )
%  C           : inv( A )      is not used with ODL nor the LS variant

%% constants and default options
mfile = 'ex413';
if exist('stopp.fil','file')  
    disp(['Delete ''stopp.fil'' to run ',mfile]);
    return;
end
if isunix()
    SPAMScatalog = '~/Matlab/SPAMS/release/mkl64/';
else
    SPAMScatalog = 'D:\Karl\DICT\matlab\SPAMS\release\win32\';
end

fmt1 = ' %20s %20s deltai=%6.3f B=%6.2f mu=%6.4f muavg=%6.4f mumse=%6.4f\n';
dataSet = 'A';
minibatch = [2000,50; 2000,100; 6000,200; 1000,500];   
parX = struct('imCat','train', 'no', 20000, ...
              'myim2colPar',struct('n',[8,8], 'i',[8,8], 'a','none', 'v',0), ...
              'transform','none', 'col',false, 'sm',true, 'norm1lim', 5);
vecSel = 'javaORMP';  
targetPSNR = 38;  
ODLvariant = 2; 
lambda0 = 0.998; 
% lambdaa     is set after input options have been checked
% D          is set after input options have been checked
verbose = 1;

%%  get the options
nofOptions = nargin;
optionNumber = 1;
fieldNumber = 1;
while (optionNumber <= nofOptions)
    if isstruct(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        sNames = fieldnames(sOptions);
        opName = sNames{fieldNumber};
        opVal = sOptions.(opName);
        % next option is next field or next (pair of) arguments
        fieldNumber = fieldNumber + 1;  % next field
        if (fieldNumber > numel(sNames)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    elseif iscell(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        opName = sOptions{fieldNumber};
        opVal = sOptions{fieldNumber+1};
        % next option is next pair in cell or next (pair of) arguments
        fieldNumber = fieldNumber + 2;  % next pair in cell
        if (fieldNumber > numel(sOptions)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    else
        opName = varargin{optionNumber};
        opVal = varargin{optionNumber+1};
        optionNumber = optionNumber + 2;  % next pair of options
    end
    % interpret opName and opVal
    if strcmpi(opName,'dataSet') || strcmpi(opName,'ds')  
        if ischar(opVal); dataSet = upper(opVal(1)); end;
    end
    if strcmpi(opName,'minibatch') || strcmpi(opName,'mb') || strcmpi(opName,'m')   
        minibatch = opVal; 
    end
    if strcmpi(opName,'parX') || strcmpi(opName,'p') 
        if isstruct(opVal); parX = opVal; end;
    end
    if strcmpi(opName,'vecSel') || strcmpi(opName,'v')  
        if ischar(opVal); vecSel = opVal; end;
    end
    if strcmpi(opName,'targetPSNR') || strcmpi(opName,'t')
        if isnumeric(opVal); targetPSNR = opVal(1); end
    end
    if strcmpi(opName,'ODLvariant') || strcmpi(opName,'o')
        if isnumeric(opVal); ODLvariant = opVal(1); end
    end
    if strcmpi(opName,'lambda0') || strcmpi(opName,'lam0') || strcmpi(opName,'l')
        if isnumeric(opVal); lambda0 = opVal(1); end
    end
    if strcmpi(opName,'lambdaa') || strcmpi(opName,'lama') || strcmpi(opName,'a')
        if isnumeric(opVal); lambdaa = opVal(1); end
    end
    if strcmpi(opName,'D0') || strcmp(opName,'D')
        if isnumeric(opVal); D = opVal; end
    end
    if strcmp(opName,'d')  % is it dataSet or initial D, type decides
        if isnumeric(opVal); D = opVal; end
        if ischar(opVal); dataSet = upper(opVal(1)); end;
    end
    %
    if strcmpi(opName,'verbose') || strcmpi(opName,'v')
        if (islogical(opVal) && opVal); verbose = 1; end;
        if isnumeric(opVal); verbose = opVal(1); end;
    end
end

if       (dataSet == 'A');   N = 8*8;      K = 256;  parXn = [8,8];    parX.col = false; 
elseif   (dataSet == 'B');   N = 12*12*3;  K = 512;  parXn = [12,12];  parX.col = true;  
elseif   (dataSet == 'C');   N = 16*16;    K = 1024; parXn = [16,16];  parX.col = false; 
else          dataSet='A';   N = 8*8;      K = 256;  parXn = [8,8];    parX.col = false; 
end
parX.myim2colPar.n = parXn;
parX.myim2colPar.i = parXn;
if ~exist('lambdaa','var')
    lambdaa = 0.95*sum(minibatch(:,1).*minibatch(:,2)); 
end

targetMSE = N * 255^2 * 10^(-targetPSNR/10);
sparseNum = floor(min(0.5*N,90)); % ORMP returns when ||w||_0 == param.sparseNum
absErrorLimit = sqrt(targetMSE);  % ORMP returns when ||r||_2 <= param.absErrorLimit

paramLasso = struct( 'mode',1, 'lambda',targetMSE, 'L',floor(0.9*N)  );
paramOMP = struct( 'eps',targetMSE,  'L',floor(0.5*N)  );

if strcmpi(vecSel, 'javaORMP')
    java_access;
else    % check access to SPAMS  (java is not needed here)
    t = version('-release');
    if (eval(t(1:4)) < 2009) || strcmpi(t,'2009a')
        res = 'mexOMP/mexLasso works only when Matlab version is greater or equal to 2009b.';
        disp(res);
        return
    end
    if exist('mexLasso.m','file')
        disp('Access to mexOMP/mexLasso.');
    elseif exist([SPAMScatalog,'mexLasso.m'],'file')
        addpath(SPAMScatalog);
    else
        res = 'Can not locate mexOMP/mexLasso on this computer.';
        disp(res);
        return
    end
end

if ~exist('D', 'var') || (size(D,1) ~= N)  || (size(D,2) ~= K)
    D = getXfromBSDS300(parX, 'no',2*K); 
    I = randperm(size(D,2));
    D = D(:, I(1:K) ); 
end
% Dn = D ./ repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
X = D;
g = sum( D.*D );     % squared norm of each x_i
A = diag(g);         % A(i,i) = W(i,i)^2
D = D ./ repmat(sqrt(g), [N 1]);      % D is normalized  
B = D .* repmat(g, [N 1]);

tab_L = ceil(sum(minibatch(:,1).*minibatch(:,2))/parX.no)+5;
res = struct(  'startTime', datestr(now()), ...
               'endTime', datestr(now()), ...
               'totalTime', -1, ...
               'inputParam', struct('mfile',mfile, 'vecSel',vecSel, 'dataSet',dataSet, ...
                          'lambda0',lambda0, 'lambdaa',lambdaa, 'targetPSNR',targetPSNR, ...
                          'targetMSE',targetMSE, 'absErrorLimit',absErrorLimit, ...
                          'dataLength',parX.no, 'ODLvariant',ODLvariant, ...
                          'sparseNum',sparseNum, 'parX',parX, 'minibatch',minibatch ), ...
               'vectorsProcessed', 0, ...
               'tvptab', zeros(1, tab_L), ...  
               'fbA', zeros(1, tab_L), ...  
               'fbB', zeros(1, tab_L), ...
               'mu', zeros(1, tab_L), ...
               'muavg', zeros(1, tab_L), ...
               'mumse', zeros(1, tab_L), ...
               'deltai', zeros(1, tab_L), ...  
               'computer', computer(), ...
               'D', D );                     % final dictioary
%           
disp([mfile,': dataSet=',dataSet,', vecSel=',vecSel,', lambda0=',num2str(lambda0),...
     ', lambdaa=',int2str(lambdaa),', targetPSNR=',num2str(targetPSNR),...
     ', ODLvariant=',int2str(ODLvariant),...
     ', Mtv=',num2str(sum(minibatch(:,1).*minibatch(:,2))*1e-6)]);

xno = size(X,2);
tvp = 0;
tab_i = 0;

tic;
for linje = 1:size(minibatch,1)
    for bno = 1:minibatch(linje,1)
        if isnan(D(1,1))
            error(['D is NaN, tvp=',int2str(tvp),', linje=',...
                   int2str(linje),', bno=',int2str(bno)]);
        end
        batchsize = minibatch(linje,2);
        if (xno+batchsize) > size(X,2)
            X = getXfromBSDS300(parX);
            xno = 0;
            if (tvp == 0)
                t = 'Initial dictionary';
                d = 0;
            else
                t = sprintf('D Mtvp=%7.3f',tvp*1e-6);
                d = dictdiff(D, res.D, 'mean', 'thabs')*180/pi;
            end
            p = dictprop(D,false);
            if verbose
                fprintf(fmt1, t, datestr(now), d, p.B, p.mu, p.muavg, p.mumse);
            end
            res.D = D;
            if tab_i < tab_L
                tab_i = tab_i + 1;
                res.tvptab(tab_i) = tvp;
                res.fbA(tab_i) = p.A;
                res.fbB(tab_i) = p.B;
                res.mu(tab_i) = p.mu;
                res.muavg(tab_i) = p.muavg;
                res.mumse(tab_i) = p.mumse;
                res.deltai(tab_i) = d;
            end
            if exist('stopp.fil','file')  % a controlled stop of the iterations
                break; 
            end
            if (xno+batchsize) > size(X,2)
                disp('  WARNING: not enough vectors in X.');
                break;
            end
        end
        I = xno+(1:batchsize);
        XI = X(:,I);
        xno = xno+batchsize;
        if strcmpi(vecSel, 'mexLasso')
            W = mexLasso(XI, D, paramLasso);
        elseif strcmpi(vecSel, 'mexOMP')
            W = mexOMP(XI, D, paramOMP);
        elseif strcmpi(vecSel, 'javaORMP')
            W = sparseapprox(XI, D, 'javaORMP', 'tae',absErrorLimit, 'tnz',sparseNum);
            maw = max(abs(W(:)));
            if (maw < 1e-6) || (maw > 1e8)
                t = 'max(abs(W)) error';
                d  = 0;
                p = dictprop(D,false);
                fprintf(fmt1, t, datestr(now), d, p.B, p.mu, p.muavg, p.mumse);
                error(['Probably error (maw=',num2str(maw),') in tvp=',int2str(tvp),', linje=',...
                      int2str(linje),', bno=',int2str(bno)]);
            end
            if sum(isnan(W(:))) > 0
                t = 'nan(W(:)) error';
                d  = 0;
                p = dictprop(D,false);
                fprintf(fmt1, t, datestr(now), d, p.B, p.mu, p.muavg, p.mumse);
                error(['Some columns in W are nan in tvp=',int2str(tvp),', linje=',...
                      int2str(linje),', bno=',int2str(bno)]);
               % w = isnan(sum(W)); I = I(w == 0); W = W(:, w == 0); % try to continue             
            end
        end
        lam = lambdafun(tvp, 'Q', lambdaa, lambda0, 1);
        lam = lam^batchsize;
        A = lam*A + full(W*W');
        B = lam*B + full(XI*W');
        if ODLvariant == 1 % seems to work well
            for j=1:K
                uj = (1/A(j,j))*(B(:,j)-D*A(:,j)) + D(:,j);
                ujn = sqrt(uj'*uj);
                D(:,j) = uj/max(ujn,1);  % normalize only large ones
            end        
        elseif ODLvariant == 2  % seems to work well
            for j=1:K
                uj = (1/A(j,j))*(B(:,j)-D*A(:,j)) + D(:,j);
                ujn = sqrt(uj'*uj);
                D(:,j) = uj/ujn; 
            end        
        elseif ODLvariant == 3  % somehow this does not seem to work well
            D = D + ( (B - D*A) ./ repmat(reshape(diag(A),1,K), [N 1]) );
            g = sqrt(sum(D.*D));
            D = D ./ repmat(g, [N 1]);
        elseif ODLvariant == 4  % and this too may give errors ???
            D = D + ( (B - D*A) ./ repmat(reshape(diag(A),1,K), [N 1]) );
            g = sqrt(sum(D.*D));
            D = D ./ repmat(g, [N 1]);
            A = A .* (g'*g);
            B = B .* repmat(g, [N 1]);
        elseif ODLvariant == 5  % (R)LS-DLA but no scaling of A (or C) and B
            D = B/A;
            g = sqrt(sum(D.*D));
            D = D ./ repmat(g, [N 1]);
        elseif ODLvariant == 6  % (R)LS-DLA and scaling of A (or C) and B
            % this should be exactly as minibatch with RLS-DLA
            D = B/A;
            g = sqrt(sum(D.*D));
            D = D ./ repmat(g, [N 1]);
            A = A .* (g'*g);
            B = B .* repmat(g, [N 1]);
        end
        tvp = tvp+batchsize;
    end
    %
    if exist('stopp.fil','file')  % a controlled stop of the iterations
        break; 
    end
end

t = sprintf('Final D Mtvp=%7.3f',tvp*1e-6);
d = dictdiff(D, res.D, 'mean', 'thabs')*180/pi;
p = dictprop(D,false);
fprintf(fmt1, t, datestr(now), d, p.B, p.mu, p.muavg, p.mumse);
res.D = D;
if tab_i < tab_L
    tab_i = tab_i + 1;
    res.tvptab(tab_i) = tvp;
    res.fbA(tab_i) = p.A;
    res.fbB(tab_i) = p.B;
    res.mu(tab_i) = p.mu;
    res.muavg(tab_i) = p.muavg;
    res.mumse(tab_i) = p.mumse;
    res.deltai(tab_i) = d;
end
res.tvptab = res.tvptab(1:tab_i);
res.fbA = res.fbA(1:tab_i);
res.fbB = res.fbB(1:tab_i);
res.mu = res.mu(1:tab_i);
res.muavg = res.muavg(1:tab_i);
res.mumse = res.mumse(1:tab_i);
res.deltai = res.deltai(1:tab_i);

res.totalTime = toc;
t=testDLplot(res, 0, 0);
res.SRC = t(9);
res.PSNR = t(10);
res.vectorsProcessed = tvp;
res.endTime = datestr(now());
%
% if ~exist('resultFile', 'var')
dstr = datestr(now());
resultFile = [mfile,'_',dstr([4:6,1,2,13,14,16,17]),'.mat'];
res.resultFile = resultFile;
save(resultFile, 'res');

return



% vector selection without sparseapprox
% jD = mpv2.SimpleMatrix(D);
% jDD = mpv2.SymmetricMatrix(K, K);
% jDD.eqInnerProductMatrix(jD);
% jMP = mpv2.MatchingPursuit(jD, jDD);
% W = zeros(K, numel(I));
% for j=1:numel(I);
%     x = X(:,I(j));
%     if sqrt(x'*x) >= absErrorLimit
%         w = jMP.vsORMP(x, sparseNum, absErrorLimit/sqrt(x'*x));
%         if (~isnan(w(1))) && (max(abs(w)) > 1e-6)
%             W(:,j) = full(w);
%         else
%             error(['Probably error in tvp=',int2str(tvp),', linje=',...
%                 int2str(linjej),', bno=',int2str(bno)]);
%         end
%     end
% end
