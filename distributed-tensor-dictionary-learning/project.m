clc;
% ex211           Show results from ex210 for L=2000 and snr=20 

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  25.06.2009  KS: made file
% Ver. 1.1  30.06.2009  KS: adapt. to changes in DictionaryLearning class
% Ver. 1.2  06.08.2009  KS: use ex210 to generate the data
% Ver. 1.3  12.04.2013  KS: use ex210 (April 2013 version) to generate the data
%----------------------------------------------------------------------

mfile = 'project';

nofTrials = 1;  % at least so many trials should be done
L = 2000;        % number of training vectors to use
snr = 20;        % snr for added noise
s = 5;           % sparseness
noIt = 200;      % number of iterations in each trial

% the file where the set of training vectors, X, is stored during design
N = 20;
K = 50;

fileNameSufix = sprintf('%1i%02i%02i.mat',s,floor(L/1000),floor(snr));

% select the methods to compare with each other and define file names
dataFiles = [
             %['func_K', fileNameSufix] % 'K' = K-SVD,   
             ['func_A', fileNameSufix] % 'A' = AK-SVD,
             %['func_T', fileNameSufix] % 'T' = K-HOSVD
             ['func_D', fileNameSufix] % 'D' = MOD,
             ['func_M', fileNameSufix] % 'M' = ILS-DLA MOD,             
             %['func_I', fileNameSufix] % 'I' = ILS-DLA MOD (java),
             %['func_U', fileNameSufix] % 'U' = T-ILS-DLA MOD,
             %['func_B', fileNameSufix] % 'B' = RLS-DLA miniBatch
             %['func_L', fileNameSufix] % 'L', 'Q', 'C', 'H' or 'E' = RLS-DLA (java),
             ];

colChar = 'brgmyck';
colName = {'Blue', 'Red', 'Green', 'Magenta', 'Yellow', 'Cyan', 'Black'};
betatab = [0.25:0.25:10, 10.5:0.5:25];
clf; 
hold on;
grid on;
x = 9.5; y = 4.5; dy = 2.5;     % for text, labels

for i=1:size(dataFiles,1);      % for selected methods for comparison
    % load or make the data
    if exist(dataFiles(i,:),'file')
        fileName = dir(dataFiles(i,:));
        disp([mfile,': use results stored in ',dataFiles(i,:),', (created ',fileName.date,').']);
        load(dataFiles(i,:));    % load res
        trialsDone = res.nofTrials;
        if res.noIt ~= noIt
            disp(['  ** OBS : noIt in file is ',int2str(res.noIt),...
                  ' while wanted (here) noIt is ',int2str(noIt),' **.']);
        end
    else
        trialsDone = 0;
    end
    
    if (nofTrials > trialsDone)
        method = dataFiles(i,6);
        res = execDL(L, N, K, snr, method, s, noIt, nofTrials-trialsDone, 0)
    end
    
    % plot the data
    yres = zeros(numel(betatab),1);
    for i1=1:numel(betatab);
        yres(i1) = sum(res.beta(:) < betatab(i1));
    end
    
    plot(betatab, yres/res.nofTrials, [colChar(i), '-']);
    h = text(x, y,['   noIt=', int2str(res.noIt),' and nofTrials=', int2str(res.nofTrials)]);
    set(h, 'BackgroundColor', [1,1,1]); 
    y = y+dy;
    I = find(yres > y*res.nofTrials);
    i1 = I(1);
    
    if ((i1>1) && (yres(i1) > yres(i1-1)))
        xp = betatab(i1-1) + (betatab(i1)-betatab(i1-1))*(y*res.nofTrials - yres(i1-1))/(yres(i1)-yres(i1-1));
    else
        xp = betatab(i1);
    end
    
    h = text(x, y,[colName{i},': ',res.method,' snr=',num2str(res.snr),' dB, L=',int2str(res.L)]);
    set(h,'BackgroundColor',[1,1,1]);
    plot([xp,x-0.5],[y,y],[colChar(i),'.-']);
    y = y+dy;
end

% plot configuration
h = text(x, y,'Dictionary learning method: ');
set(h,'BackgroundColor',[1,1,1]);
title({'Number of dictionary atoms identified, average over the trials.';
    ['Plot generated ',datestr(now()),'. (ex211.m ver 1.3, by Karl Skretting, UiS).']} );
xlabel('Limit for positive identification, \beta_l_i_m [degrees].');
ylabel('Number of identified atoms');
print( gcf, '-depsc2', 'ex211.eps' );
disp('Printed figure as: ex211.eps');
print( gcf, '-dpng', '-r80', 'ex211.png' );
disp('Printed figure as: ex211.png');

return