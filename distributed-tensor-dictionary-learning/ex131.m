% ex131           RLS-DLA with adaptive increasing lambda. 
%   1   - AR(1) signal, s=4, N=16, K=32
%    3  - experiment with RLS-DLA and increasing lambda
%     1 - number 1, plot average for many trials 
%
% To generate new data, make sure to delete or rename 'ex131.mat',
% if this mat-file exists the stored results will be used.
% New results are generated using the adaptive schemes for increasing
% lambda as given by workspace variable 'par', or default if not given.
% 'par' is a cell array containing structs with fields
%  .lamM  - the method, 'L', 'Q', 'C', 'H' or 'E'
%  .lam0  - the initial value for lambda (typically 0.98 <= lam0 < 1)
%  .a     - the parameter a given as ratio to total number of iterations
%           for 'L', 'Q' and 'C' typically in range 0.8-1, 
%           for 'H' and 'E' typically in range 0.05-0.25
% Workspace variables 'par', 'noIT' and 'noEX' (number of trials) are 
% only used when new data is generated. 'makePlot' specify what kind of 
% plot to make, it can be 'bw', 'col', or 'none'.
%
% example:
% par = cell(6,1);
% par{1} = struct('lamM','L','lam0',0.99,'a',0.9);
% par{2} = struct('lamM','Q','lam0',0.99,'a',1);
% par{3} = struct('lamM','C','lam0',0.99,'a',1);
% par{4} = struct('lamM','H','lam0',0.99,'a',0.05);
% par{5} = struct('lamM','E','lam0',0.98,'a',0.1);
% par{6} = struct('lamM','L','lam0',0.985,'a',0.9);
% makePlot='col'; noIT = 400; noEX = 10; 
% ex131;

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  24.04.2009  KS: test started
% Ver. 1.1  07.05.2009  KS: make figure in color or black and white
% Ver. 1.2  23.06.2009  KS: use in dle (Dictionary Learning Experiment)
% Ver. 1.3  30.06.2009  KS: adapt. to changes in DictionaryLearning class
%----------------------------------------------------------------------

Mfile = 'ex131';
% the file where the set of training vectors, X, is stored during design
DataFile = 'dataXforAR1.mat';    
s = 4;  
K = 32;

% if ~exist('ResultFile','var')
ResultFile = [Mfile,'.mat'];
% end
if ~exist('noIT','var')
    noIT = 400;
end
if ~exist('noEX','var')
    noEX = 10; 
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
    disp(' ');
    if ~exist('makePlot','var')
        makePlot = 'none';
    end
    if (~exist('par','var') || ~iscell(par) || ~isstruct(par{1}))
        disp('Use default set of adaptive schemes for lambda, variable <par>.');
        par = cell(6,1);
        par{1} = struct('lamM','L','lam0',0.99,'a',0.9);
        par{2} = struct('lamM','Q','lam0',0.99,'a',1);
        par{3} = struct('lamM','C','lam0',0.99,'a',1);
        par{4} = struct('lamM','H','lam0',0.99,'a',0.05);
        par{5} = struct('lamM','E','lam0',0.98,'a',0.1);
        par{6} = struct('lamM','L','lam0',0.985,'a',0.9);
    end

    disp([Mfile,': learn ',int2str(N),'x',int2str(K),' dictionaries for training vectors in ',DataFile]);
    disp(['Generate new results for ',int2str(noEX),' trials, each with ',int2str(noIT),' iterations.']);
    disp(['Each trial used ',int2str(numel(par)),' adaptive schemes for lambda.']);

    res = cell(numel(par),noEX);
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
            jDicLea.setLambda( par{i}.lamM, par{i}.lam0, 1.0, (noIT*L)*par{i}.a );

            tic;
            jDicLea.rlsdla( X(:), noIT );
            t = toc;
            snrTab = jDicLea.getSnrTab();

            disp(['Trial ',int2str(exno),', set-',int2str(i),' time = ',num2str(t),...
                ', ',int2str(noIT),' iterations.',...
                'lamM = ',par{i}.lamM,', lam0 = ',num2str(par{i}.lam0),...
                ', ratio a/noIT = ',num2str(par{i}.a) ]);
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
            newSNR = 10*log10(sumXX/sum(sum(R.*R)));
            I = floor(noIT*[25,50,75,100]/100);
            disp(['SNR(',int2str(I(1)),') is ',num2str(snrTab(I(1))),...
                ', SNR(',int2str(I(2)),') is ',num2str(snrTab(I(2))),...
                ', SNR(',int2str(I(3)),') is ',num2str(snrTab(I(3))),...
                ', SNR(',int2str(I(4)),') is ',num2str(snrTab(I(4))),...
                ', Actual SNR is now ',num2str( newSNR )]);
            res{i,exno} = struct('exno',exno,'pset',i,'snr', snrTab ,'time',t,...
                'D',D,'newSNR',newSNR);
            % 
            timeleft = (now()-timestart)*((1+noEX-exno)*numel(par)-i)/((exno-1)*numel(par)+i);
            disp(['Estimated finish time is ',datestr(now()+timeleft)]);
        end
        save([Mfile,'.mat'], 'res', 'par');
        X = X(:,randperm(L));
    end
end

noEX = size(res,2);
noIT = numel(res{1}.snr);
% ymat is the average for each setup, snrend is end
snrend = zeros(size(res));
snrnew = zeros(size(res));
ymat = zeros(noIT, size(res,1));
for exno=1:size(res,2)
    for i=1:size(res,1)
        ymat(:,i) = ymat(:,i)+res{i,exno}.snr;
        snrend(i,exno) = res{i,exno}.snr(end);
        snrnew(i,exno) = res{i,exno}.newSNR;
    end
end
ymat = ymat/noEX;

% legend text for each line
legtext = cell(numel(par),1);
for i=1:size(par,1)
    t = ['RLS-DLA, \lambda: ',par{i}.lamM,' ',...
            num2str(par{i}.lam0),' -> 1.0 (a=',num2str(par{i}.a),')' ];
    legtext{i} = [t,', SNR(',int2str(size(ymat,1)),')=',sprintf('%2.2f',ymat(end,i))];
end


if (strcmpi(makePlot,'bw') || strcmpi(makePlot,'col'))
    mark = 'ov*sp>hx+<ov*spv<>*hx+ov';
    if strcmpi(makePlot,'bw')
        col = 'kkkkkkkkkkkkkkkkkk';
    else
        col = 'bgrcmkbgrckbgrcmk';  
    end
    xrange = 1:size(ymat,1);
    mpnt = floor(noIT*[5,20,35,50,65,80,95]/100);  % markers on these points
    figure(1);clf;hold on;grid on;
    I = 1:size(ymat,2);
    for i = I;
        h = plot(xrange, ymat(:,i), [col(i),'-']);
        set(h,'LineWidth',1.0);
        h = plot(mpnt, ymat(mpnt,i), [col(i),mark(i)]);
    end
    V = axis(); V(3:4)=[17.25,18.25]; axis(V);
    ypos = 17.34; yinc = 0.07;
    x1 = floor(noIT*25/100);
    x2 = floor(noIT*35/100);
    x3 = floor(noIT*37/100);
    [temp,J] = sort(ymat(x1,I));
    for j=J
        plot([x1,x2], [ymat(x1,j),ypos], [col(j),mark(j),'-']);
        h = text(x3, ypos, legtext{j});
        set(h,'BackgroundColor',[1,1,1]);
        set(h,'Color',col(j));
        ypos = ypos+yinc;
    end
    title({['RLS-DLA and adaptive \lambda for AR(1) signal, average over ',int2str(noEX),' trials.'];
           ['Plot generated ',datestr(now()),'. (',Mfile,'.m ver 1.3, by Karl Skretting, UiS).']} );
    xlabel('Iteration number');
    ylabel('SNR in dB');
    print('-f1','-depsc2',[Mfile,'.eps']);
    disp(['Printed figure 1 as: ',[Mfile,'.eps']]);
end


