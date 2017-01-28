% ex111           Compare results for ILS-DLA (MOD), K-SVD and RLS-DLA. 
%                 RLS-DLA is here used without forgetting factor lambda.
% An AR(1) signal is used here. Sparseness is s=4. 
% Dictionary size is N=16, K=32
% The java class mpv2 is used for ILS-DLA and RLS-DLA as well as in sparse
% approximation (javaORMP)
%
% To generate new data, make sure to delete or rename 'ex111.mat',
% if this mat-file exists the stored results will be used.
% Workspace variables 'noIT' and 'noEX' (number of trials) are only used
% when new data is generated. 'makePlot' specify what kind of plot to make,
% it can be 'bw', 'col', or 'none'.
% (To generate new data takes approx. 15 minutes per 1000 iterations)
%
% example:
% clear all; makePlot='col'; noIT = 200; noEX = 5; ex111;  

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  22.04.2009  KS: test started
% Ver. 1.1  07.05.2009  KS: make figure in color or black and white
% Ver. 1.2  23.06.2009  KS: use in dle (Dictionary Learning Experiment)
% Ver. 1.3  30.06.2009  KS: adapt to changes in DictionaryLearning class
% Ver. 1.4  11.04.2011  KS: use sparseapprox with javaORMP
%----------------------------------------------------------------------

Mfile = 'ex111';
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
    disp([Mfile,': learn ',int2str(N),'x',int2str(K),' dictionaries for training vectors in ',DataFile]);
    disp(['Generate new results for ',int2str(noEX),' trials, each with ',int2str(noIT),' iterations.']);
    res = cell(3,noEX);
    javaAccess;
    timestart = now();
    for exno = 1:noEX
        disp(' ');
        disp(['Start trial number ',int2str(exno)]);
        D0 = X(:,1500+(1:K));   % no normalization here !
        jD0 = mpv2.SimpleMatrix(D0);

        % first ILS-DLA
        jDicLea  = mpv2.DictionaryLearning(jD0, 1);
        jDicLea.setORMP(int32(s), 1e-6, 1e-6);
        tic;
        jDicLea.ilsdla( X(:), noIT );
        t = toc;
        snrTab = jDicLea.getSnrTab();
        %  use D to find new SNR
        jD =  jDicLea.getDictionary();
        D = reshape(jD.getAll(), N, K);
        W = sparseapprox(X, D, 'javaORMP', 'tnz',s);
        R = X - D*W;
        newSNR = 10*log10(sumXX/sum(sum(R.*R)));
        middleindex = floor(noIT/2);
        disp(['Trial ',int2str(exno),', ILS-DLA time = ',num2str(t),...
            ', SNR(',int2str(middleindex),') is ',num2str(snrTab(middleindex)),...
            ', SNR(end) is ',num2str(snrTab(end)),...
            ', Actual SNR is now ',num2str( newSNR )]);
        res{1,exno} = struct('exno',exno, 'snr', snrTab , 'time',t,...
            'D',D, 'newSNR',newSNR);

        % if 0 % Second K-SVD
        D = D0;
        snrTab = zeros(noIT,1);
        tic
        for it = 1:noIT
            if 0 % chech dictionary
                D = dictnormalize(D);
                DD = D'*D;
                DD(abs(DD)>1)=1;
                Ang = acos(abs(DD))*180/pi + 360*eye(K);
                [amin, aind] = min(min(Ang));
                while (amin < 1.0)
                    disp(['it-',int2str(it),...
                        ', Smallest angle between dictioanry vectors is ',num2str(amin)]);
                    d = X(:,2000+floor(rand(1)*500))+0.01*randn(N,1);
                    d = d/sqrt(d'*d);
                    D(:,aind) = d;
                    DD = D'*D;
                    DD(abs(DD)>1)=1;
                    Ang = acos(abs(DD))*180/pi + 360*eye(K);
                    [amin, aind] = min(min(Ang));
                end
            end
            % find weights, using dictionary D
            W = sparseapprox(X, D, 'javaORMP', 'tnz',s);
            % K-SVD
            for k=1:K
                R = X - D*W;
                I = find(W(k,:));
                Ri = R(:,I) + D(:,k)*W(k,I);
                [U,S,V] = svds(Ri,1,'L');
                D(:,k) = U;
                W(k,I) = S*V';
            end
            R = X - D*W; noIT = numel(res{1}.snr);

            snrTab(it) = 10*log10(sumXX/sum(sum(R.*R)));
        end
        t = toc;
        %  use D to find new SNR
        W = sparseapprox(X, D, 'javaORMP', 'tnz',s);
        R = X - D*W;
        newSNR = 10*log10(sumXX/sum(sum(R.*R)));
        middleindex = floor(noIT/2);
        disp(['Trial ',int2str(exno),', K-SVD time = ',num2str(t),...
            ', SNR(',int2str(middleindex),') is ',num2str(snrTab(middleindex)),...
            ', SNR(end) is ',num2str(snrTab(end)),...
            ', Actual SNR is now ',num2str( newSNR )]);
        res{2,exno} = struct('exno',exno, 'snr', snrTab , 'time',t,...
            'D',D, 'newSNR',newSNR);
        % end

        % and third, RLS-DLA
        jDicLea  = mpv2.DictionaryLearning(jD0, 1);
        jDicLea.setLambda('1', 1.0, 1.0, 100);  % not needed as this is default.
        jDicLea.setORMP(int32(s), 1e-4, 1e-4);
        jDicLea.setLoggOn();
        jDicLea.setVerbose(0);
        jDicLea.setParamI1(16);
        tic;
        jDicLea.rlsdla( X(:), int32(noIT) ); 
        t = toc;
        snrTab = jDicLea.getSnrTab();
        %  use D to find new SNR
        jD =  jDicLea.getDictionary();
        D = reshape(jD.getAll(), N, K);
        if isnan(D(1,1))
            disp('D is NaN.');
            W = zeros(K,L);
            R = X;
        else
            W = sparseapprox(X, D, 'javaORMP', 'tnz',s);
            R = X - D*W;
        end
        newSNR = 10*log10(sumXX/sum(sum(R.*R)));
        middleindex = floor(noIT/2);
        disp(['Trial ',int2str(exno),', RLS-DLA time = ',num2str(t),...
            ', SNR(',int2str(middleindex),') is ',num2str(snrTab(middleindex)),...
            ', SNR(end) is ',num2str(snrTab(end)),...
            ', Actual SNR is now ',num2str( newSNR )]);
        res{3,exno} = struct('exno',exno, 'snr', snrTab , 'time',t,...
            'D',D, 'newSNR',newSNR);
        %
        timeleft = (noEX-exno)*(now()-timestart)/exno;
        disp(['Estimated finish time is ',datestr(now()+timeleft)]);
        X = X(:,randperm(L));
    end
    save(ResultFile, 'res');  % noEX and noIT stored as sizes in res
end

noEX = size(res,2);
noIT = numel(res{1}.snr);
% ymat is the average for each setup.
ymat = zeros(numel(res{1,1}.snr), size(res,1));
for exno=1:size(res,2)
    for i=1:size(res,1)
        ymat(:,i) = ymat(:,i)+res{i,exno}.snr;
    end
end
ymat = ymat/size(res,2);
% legend text for each line
legtext = cell(3,1);
legtext{1} = ['ILS-DLA, SNR(',int2str(size(ymat,1)),')=',sprintf('%2.2f',ymat(end,1))]; 
legtext{2} = ['K-SVD, SNR(',int2str(size(ymat,1)),')=',sprintf('%2.2f',ymat(end,2))]; 
legtext{3} = ['RLS-DLA \lambda=1, SNR(',int2str(size(ymat,1)),')=',sprintf('%2.2f',ymat(end,3))]; 

if (strcmpi(makePlot,'bw') || strcmpi(makePlot,'col'))
    mark = 'ovs*p>h<x+ovov<*spv<>*hx+ov';
    epsfile = [Mfile,'.eps'];
    if strcmpi(makePlot,'bw')
        col = 'kkkkkkkkkkkkkkkkkk';
    else
        % col = 'rbkcgmrgbcmk';
        col = 'bgrcmkbgrcmkbgrcmk';  
    end
    xrange = 1:size(ymat,1);
    mpnt = [25,75,125,175];  % markers on these points
    figure(1);clf;hold on;grid on;
    I = 1:3;
    for i = I;
        h = plot(xrange, ymat(:,i), [col(i),'-']);
        set(h,'LineWidth',1.0);
        h = plot(mpnt, ymat(mpnt,i), [col(i),mark(i)]);
    end
    V = axis(); V(3:4)=[17,18]; axis(V);
    [temp,J] = sort(ymat(60,I));
    ypos = 17.3; yinc = 0.1;
    for j=J
        plot([60,90], [ymat(60,I(j)),ypos], [col(I(j)),mark(I(j)),'-']);
        h = text(95, ypos, legtext{I(j)});
        set(h,'BackgroundColor',[1,1,1]);
        set(h,'Color',col(I(j)));
        ypos = ypos+yinc;
    end
    % title(['dltest10 (',datestr(now()),') RLS-DLA for AR(1) signal.']);
    title({['ILS-DLA, K-SVD and RLS-DLA for AR(1) signal, average over ',int2str(noEX),' trials.'];
           ['Plot generated ',datestr(now()),'. (',Mfile,'.m ver 1.3, by Karl Skretting, UiS).']} );
    xlabel('Iteration number');
    ylabel('SNR in dB');
    print('-f1','-depsc2',epsfile);
    disp(['Printed figure 1 as: ',epsfile]);
end


