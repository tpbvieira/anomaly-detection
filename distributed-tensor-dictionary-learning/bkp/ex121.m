% ex121           RLS-DLA with different fixed values of lambda. 
%   1   - AR(1) signal, s=4, N=16, K=32
%    2  - experiment with RLS-DLA and fixed lambda
%     1 - number 1, 
%
% To generate new data, make sure to delete or rename 'ex121.mat',
% if this mat-file exists the stored results will be used.
% New results are generated using the fixed values of lamtab below.
% Workspace variables 'lamtab', 'noIT' and 'noEX' (number of trials) are 
% only used when new data is generated. 'makePlot' specify what kind of 
% plot to make, it can be 'bw', 'col', or 'none'.
%
% example:
% lamtab = [0.9999, 0.9998, 0.9995, 0.9990, 0.9950];
% makePlot='col'; noIT = 200; noEX = 5; ex121;

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  21.04.2009  KS: test started
% Ver. 1.1  23.06.2009  KS: use in dle (Dictionary Learning Experiment)
% Ver. 1.2  30.06.2009  KS: adapt to changes in DictionaryLearning class
%----------------------------------------------------------------------

Mfile = 'ex121';
% the file where the set of training vectors, X, is stored during design
DataFile = 'dataXforAR1.mat';    
s = 4;  
K = 32;

% if ~exist('ResultFile','var')
ResultFile = [Mfile,'.mat'];
% end
if ~exist('noIT','var')
    noIT = 200;
end
if ~exist('noEX','var')
    noEX = 5; 
end

load(DataFile);   % henter X og S og noen flere felt
N = size(X,1);
L = size(X,2);
disp(' ');
sumXX = sum(sum(X.*X));

if exist(ResultFile,'file')
    d = dir(ResultFile);
    disp([Mfile,': use results stored in ',ResultFile,', (created ',d.date,').']);
    load(ResultFile);
    if ~exist('makePlot','var')
        makePlot = 'col';
    end
else
    if ~exist('makePlot','var')
        makePlot = 'none';
    end
    if ~exist('lamtab','var')
        lamtab = [0.9999, 0.9998, 0.9995, 0.9990, 0.9950];
    end
    % the different sets of parameters to use must be like this for fixed lambda
    par = cell(numel(lamtab),1);
    for i=1:numel(lamtab)
        par{i} = struct('noIT',noIT,'lamM','L','lam0',lamtab(i),'lam1',lamtab(i),'lamP',noIT*L);
    end

    disp([Mfile,': learn ',int2str(N),'x',int2str(K),' dictionaries for training vectors in ',DataFile]);
    disp(['Generate new results for ',int2str(noEX),' trials, each with ',int2str(noIT),' iterations.']);
    disp(['Each trial used ',int2str(numel(lamtab)),' fixed values of lambda.']);
    res = cell(numel(lamtab),noEX);

    java_access;
    timestart = now();
    for exno = 1:noEX
        disp(' ');
        disp(['Start trial number ',int2str(exno)]);
        D0 = X(:,1500+(1:K));   % no normalization here
        jD0 = mpv2.SimpleMatrix(D0);
        for i=1:numel(par)
            %
            jDicLea  = mpv2.DictionaryLearning(jD0, 1);
            jDicLea.setORMP(int32(s), 1e-6, 1e-6);
            jDicLea.setLambda(par{i}.lamM, par{i}.lam0, par{i}.lam1, par{i}.lamP);

            tic;
            jDicLea.rlsdla( X(:), par{i}.noIT );
            t = toc;
            snrTab = jDicLea.getSnrTab();

            disp(['Trial ',int2str(exno),', set-',int2str(i),' time = ',num2str(t),...
                ', ',int2str(par{i}.noIT),' iterations.',...
                ', lamM = ',par{i}.lamM,', lam0 = ',num2str(par{i}.lam0),', lam1 = ',num2str(par{i}.lam1) ]);
            %  use D to find new SNR
            jD =  jDicLea.getDictionary();
            D = reshape(jD.getAll(), N, K);
            jDD = mpv2.SymmetricMatrix(K,K);
            jDD.eqInnerProductMatrix(jD);
            jMP = mpv2.MatchingPursuit(jD,jDD);
            W = zeros(K,L);
            for j=1:L
                W(:,j) = jMP.vsORMP(X(:,j), int32(s));
            end
            R = X - D*W;
            newSNR = 10*log10(sum(sum(X.*X))/sum(sum(R.*R)));
            disp(['SNR(end) is ',num2str(snrTab(end)),...
                ', Actual SNR is now ',num2str( newSNR )]);
            res{i,exno} = struct('exno',exno,'pset',i,'snr', snrTab ,'time',t,...
                'D',D,'newSNR',newSNR);
            %
            timeleft = (now()-timestart)*((1+noEX-exno)*numel(par)-i)/((exno-1)*numel(par)+i);
            disp(['Estimated finish time is ',datestr(now()+timeleft)]);
        end
        save(ResultFile, 'res', 'par','lamtab');
        X = X(:,randperm(L));
    end
end

noEX = size(res,2);
noIT = numel(res{1}.snr);
% ymat is the average for each setup, snrend is end
snrend = zeros(size(res));
snrnew = zeros(size(res));
ymat = zeros(numel(res{1,1}.snr), size(res,1));
for exno=1:size(res,2)
    for i=1:size(res,1)
        ymat(:,i) = ymat(:,i)+res{i,exno}.snr;
        snrend(i,exno) = res{i,exno}.snr(end);
        snrnew(i,exno) = res{i,exno}.newSNR;
    end
end
ymat = ymat/size(res,2);

% legend text for each line
legtext = cell(size(res,1),1);
for i=1:size(res,1)
    pset = res{i,1}.pset;
    if (par{pset}.lam0 == par{pset}.lam1)
        t = ['RLS, \lambda = ',sprintf('%1.5f',par{pset}.lam1)];
    else
        t = ['RLS, \lambda = ',par{pset}.lamM,'-',...
            num2str(par{pset}.lam0),'-',num2str(par{pset}.lam1) ];
    end
    legtext{i} = [t,', SNR(',int2str(size(ymat,1)),')=',sprintf('%2.2f',ymat(end,i))];
end

if (strcmpi(makePlot,'bw') || strcmpi(makePlot,'col'))
    mark = 'ov*sp>hx+<ov*spv<>*hx+ov';
    epsfile = [Mfile,'.eps'];
    if strcmpi(makePlot,'bw')
        col = 'kkkkkkkkkkkkkkkkkk';
    else
        col = 'kbgrcmkmbgrckbgrcmk';  
    end
    xrange = 1:size(ymat,1);
    mpnt = 25:50:size(ymat,1);  % markers on these points
    figure(1);clf;hold on;grid on;
    I = 1:numel(lamtab);
    for i = I;
        h = plot(xrange, ymat(:,i), [col(i),'-']);
        set(h,'LineWidth',1.0);
        h = plot(mpnt, ymat(mpnt,i), [col(i),mark(i)]);
    end
    V = axis(); V(3:4)=[17,18]; axis(V);
    [temp,J] = sort(ymat(40,I));
    ypos = 17.04; yinc = 0.06;
    for j=J
        plot([40,80], [ymat(40,I(j)),ypos], [col(I(j)),mark(I(j)),'-']);
        h = text(85, ypos, legtext{I(j)});
        set(h,'BackgroundColor',[1,1,1]);
        set(h,'Color',col(I(j)));
        ypos = ypos+yinc;
    end
    % title(['dltest09 (',datestr(now()),') RLS-DLA for AR(1) signal.']);
    title({['RLS-DLA for AR(1) signal using fixed \lambda, average over ',int2str(noEX),' trials.'];
           ['Plot generated ',datestr(now()),'. (',Mfile,'.m ver 1.2, by Karl Skretting, UiS).']} );
    xlabel('Iteration number');
    ylabel('SNR in dB');
    print('-f1','-depsc2',epsfile);
    disp(['Printed figure 1 as: ',epsfile]);
end


