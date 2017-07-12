function res = ex312(transform, K, targetPSNR, noIT, testLena, verbose)
% ex312           Training of dictionary for images using RLS-DLA 
%                 In each iteration 12000 training vectors are made
% from 8 images. A new training set is generated in each iteration.
% Each 8x8 image block, or 8x8 coefficient block, are made
% into a column vector of length N = 64. 
% The results are saved in 'ex312mmmddhhmm.mat'.
% 
% use:
%  res = ex312(transform, K, targetPSNR, noIT, testLena, verbose)
%-------------------------------------------------------------------------
% arguments:
%   transform   as in dataXimage.m  'none', 'dct', 'lot', 'elt', 'jp2', 'ks2'
%   K           number og vectors in dictionary
%   targetPSNR  target Peak Signal to Noise Ratio, should be >= 30
%               R = X - D*W;  % not an image (R ~= A - Ar)
%               sumRR = sum(sum(R.*R));
%               PSNR = 10*log10( numel(R)*255^2 / sumRR )
%   noIT        number of iterations through each training set X,
%               each of 12000 vectors
%   testLena    0 or 1, default 0
%   verbose     0 or 1, default 0.
%-------------------------------------------------------------------------
% example:
%  res = ex312('none',128,35,50);       % a simple example
%  res = ex312('m79',440,38,200,1,1);   % another example
%  res = ex312('many');     % adjust m-file to learn many dictionaries                            

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  17.08.2009  KS: test started (based on ex311.m)
% Ver. 1.1  18.08.2009  KS: try to make function run faster 
% Ver. 1.2  29.10.2009  KS: testLena part is mainly in sparseImage
% Ver. 1.3  02.11.2009  KS: use ex314 to plot SNR during training
% Ver. 1.4  19.01.2010  KS: use ex321 to plot SNR during training and
%                       getXfrom8images (myim2col) instead of dataXimage.
% Ver. 1.5  09.08.2011  KS: Simplified a little bit
%----------------------------------------------------------------------

mfile = 'ex312';

if ((nargin == 1) && strcmpi(transform,'many'))
    K = 440;   % 128, 440 (DC excluded), 512
    noit = 400;
    target = 36;
    for i = 1:5
        ex312( 'm79', K, target, noit, 0, 0);
        ex312('none', K, target, noit, 0, 0);
    end
    res = 'done';
    return;
end

if (nargin < 3)
   error([mfile,': wrong number of arguments, see help.']);
end
if (nargin < 4); noIT = 500; end;
if (nargin < 5); testLena = 0; end;
if (nargin < 6); verbose = 0; end;

maxS = 40;
N = 64;

% limit for absolute error, ORMP end when ||r|| < maxError
% PSNR = 10*log10( numel(.)*255^2 / sum( ||r_i||^2 ) )
% setting maxError, the average error will be smaller
maxError = 2*sqrt(N)*255*10^(-targetPSNR/20);  
relLim = 1e-6;

X = getXfrom8images('t',transform, 'getFixedSet',1, 'v',1);
[N,L] = size(X);
%
disp(' ');
disp([mfile,': start RLS-DLA training for images, ',datestr(now()),...
    ', N=64, K=',int2str(K),', L=',int2str(L),...
    ', target PSNR = ',num2str(targetPSNR) ]);
% always select the DC vector, thus remove DC here
if strcmpi(transform,'none')
    X = X - ones(64,1)*mean(X);
else
    X(1,:) = zeros(1,L);
end
%
% for many training vectors the DC element will be enough
% the dictionary is trained for the rest
xsquared = sum(X.*X);   % this is ||r||^2 after DC is selected
I = find(xsquared > (maxError^2));  % for the rest DC is enough
rr = sum(xsquared) - sum(xsquared(I));
I = I(randperm(numel(I)));  % permute the elements of I

% initial dictionary, the K random training vectors
D0 = dictnormalize( X(:,I(1:K)) );
I = I((K+1):end);  % indexes for the rest

java_access;
timestart = now();
jD0 = mpv2.SimpleMatrix(D0);
jDicLea  = mpv2.DictionaryLearning(jD0, 1);
jDicLea.setORMP(int32(maxS), relLim, maxError);
jDicLea.setLambda('C', 0.99, 1.00, 0.75*numel(I)*noIT);
jDicLea.setVerbose(0);

tabPSNR = zeros(noIT,1);
tabNNZ = zeros(noIT,1);  % number of non-zeros
for iteration = 1:noIT
    %
    tic;
    % training one vector at each call to Java
    nnz = L;
    for i = I
        jDicLea.rlsdla1( X(:,i) );
        rr = rr + jDicLea.getSumrr();
        w = jDicLea.getWeights();
        nnz = nnz + sum(w~=0);
    end
    tabNNZ(iteration) = nnz;
    % train a batch of vectors in one call to Java (no results)
    %    jDicLea.rlsdla( reshape(X(:,I),N*numel(I),1), 1 );
    %    rr = rr + jDicLea.getSumAllrrTab();
    %
    PSNR = 10*log10( (N*L*255^2)/rr );
    tabPSNR(iteration) = PSNR;
    if ((rem(iteration,5) == 0) || (iteration < 5))
        disp([mfile,': transform = ',transform,...
            ' iteration ',num2str(iteration),' of ',int2str(noIT),...
            ' (nnz = ',int2str(tabNNZ(iteration)),')',...
            ', maxError = ',num2str(maxError,5),...
            ' actual PSNR = ',num2str(PSNR) ]);
        timeUsed = now() - timestart;  
        timeleft = (noIT-iteration) * timeUsed/iteration;
        disp(['Estimated finish time is ',datestr(now()+timeleft)]);
    end
    %
    % we may want to adjust maxError
    factor = 1 + 0.2*((1-iteration/noIT)^2)*min(abs(targetPSNR-PSNR),2);
    if ((PSNR > targetPSNR) && (numel(I) > (L/50)))
        % we want to select fewer, i.e. increase maxError
        % but not select too few either
        maxError = maxError*factor;
    else
        % we want to select more, i.e. decrease maxError
        maxError = maxError/factor;
    end
    jDicLea.setORMP(int32(maxS), relLim, maxError);
    %
    % new set of training data in each iteration !
    if (iteration < noIT)
        X = getXfrom8images('t',transform);
        % always select the DC vector, thus remove DC here
        if strcmpi(transform,'none')
            X = X - ones(64,1)*mean(X);
        else
            X(1,:) = zeros(1,L);
        end
        xsquared = sum(X.*X);   % this is ||r||^2 after DC is selected
        I = find(xsquared > (maxError^2));  % for the rest DC is enough
        rr = sum(xsquared) - sum(xsquared(I));
        I = I(randperm(numel(I)));  % permute the elements of I
    end
end
%
jD =  jDicLea.getDictionary();
D = reshape(jD.getAll(), N, K);
dstr = datestr(now());
ResultFile = [mfile,dstr([4:6,1,2,13,14,16,17]),'.mat'];
if (verbose >= 0); disp(['Save results in ',ResultFile]); end;
timeUsed = (now() - timestart)*(24*60);  % in minutes
save(ResultFile, 'D','tabNNZ','tabPSNR','ResultFile','targetPSNR',...
    'timeUsed','N','K','L','transform');
%

res = struct('D',D,...
             'tabNNZ',tabNNZ,...
             'tabPSNR',tabPSNR,...
             'ResultFile',ResultFile,...
             'targetPSNR',targetPSNR,...
             'timeUsed',timeUsed, ....
             'N',N,'K',K,'L',L,...
             'transform',transform );
%

% display some properties for the results
ex31prop(ResultFile);
    
if testLena
    % sparse representation of lena using trained dictionary
    targetPSNRtab = [32, 34, 36, 38];
    r2 = cell(size(targetPSNRtab));
    for i = 1:numel(targetPSNRtab)
        r2{i} = imageapprox(double(imread('lena.bmp'))-128, ...
            'Transform',res.transform, ...
            'Dictionary',res.D, ...
            'targetPSNR',targetPSNRtab(i), ...
            'peak',255, ...
            'delta',0, ...
            'verbose', 1);
    end
    res.r2 = r2;
end

return;
