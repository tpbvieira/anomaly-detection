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
% Ver. 2.3  00.02.2017  Version of Thiago Vieira adding tersor-based
% methods
%----------------------------------------------------------------------
clc;
set(0,'RecursionLimit',1200)
mfile = 'project';
%
s = 5;           % sparseness
snr = 20;        % snr for added noise
L = 2000;        % number of training vectors to use
nofTrials = 5;   % at least so many trials should be done
noIt = 100;      % number of iterations in each trial
N = 80;
K = 200;
M1 = 10;
M2 = 8;
N1 = 20;
N2 = 10;

% select the methods to compare and define file names
fileNameInfo = sprintf('%1i_%li_%li_%li_%li_%li',s,snr,L,nofTrials,noIt,N*K);
fileNameSufix = sprintf('%s.mat',fileNameInfo);
dataFiles = [
             ['K', fileNameSufix] % 'K' = K-SVD,   
             %['func_A', fileNameSufix] % 'A' = AK-SVD,
             ['T', fileNameSufix] % 'T' = K-HOSVD
             ['D', fileNameSufix] % 'D' = MOD,
             ['O', fileNameSufix] % 'O' = T-MOD,
             %['M', fileNameSufix] % 'M' = ILS-DLA MOD,             
             %['I', fileNameSufix] % 'I' = ILS-DLA MOD (java),             
             %['B', fileNameSufix] % 'B' = RLS-DLA miniBatch
             ['L', fileNameSufix] % 'L', 'Q', 'C', 'H' or 'E' = RLS-DLA (java),
             ];

% plot configuration
title({'Number of dictionary atoms identified per degrees.'; 'Elements: ';N*K});
ylabel('Number of identified atoms.');
xlabel('Required degrees for identification.');
epsName = sprintf('%1i_%li_%li_%li_%li_%li.eps',s,snr,L,nofTrials,noIt,N*K);
pngName = sprintf('%1i_%li_%li_%li_%li_%li.png',s,snr,L,nofTrials,noIt,N*K);
colChar = 'brgmyck';
colName = {'Blue', 'Red', 'Green', 'Magenta', 'Yellow', 'Cyan', 'Black'};
degreesRates = [0.25:0.25:10, 10.5:0.5:25];
clf; 
hold on;
grid on;
x = 8; y = 7; yPadding = 10;

% for selected methods for comparison
for i=1:size(dataFiles,1);  
    
    % load or make the data
    if exist(dataFiles(i,:),'file')
        fileName = dir(dataFiles(i,:));
        disp([mfile,': results of ',dataFiles(i,:),', ',fileName.date,'.']);
        load(dataFiles(i,:));                                               % load privious results
        trialsDone = result.nofTrials;
        if result.noIt ~= noIt
            disp(['  ** OBS : noIt in file is ',int2str(result.noIt),...
                  ' while wanted (here) noIt is ',int2str(noIt),' **.']);
        end
    else
        trialsDone = 0;
    end
    
    % execute trials for atoms identification
    if (nofTrials > trialsDone)
        method = dataFiles(i,1);
        result = execDL(L, N, K, M1, M2, N1, N2, snr, method, s, noIt, nofTrials-trialsDone, 0)
    end
    
    % prepare data
    yPos = zeros(size(K,1),1);
    for i1=1:numel(degreesRates);
        yPos(i1) = sum(result.beta(:) < degreesRates(i1));
    end
    yPos = yPos/result.nofTrials; % simple mean by trials
    
    % update plot for the current method
    plot(degreesRates, yPos, [colChar(i), '-']);
    legend = text(x, y,['   noIt=', int2str(result.noIt),' and nofTrials=', int2str(result.nofTrials)]);
    set(legend, 'BackgroundColor', [1,1,1]); 
    y = y+yPadding;
    I = find(yPos > y*result.nofTrials);
    i1 = I(1);
    
    if ((i1>1) && (yPos(i1) > yPos(i1-1)))
        xp = degreesRates(i1-1) + (degreesRates(i1)-degreesRates(i1-1))*(y*result.nofTrials - yPos(i1-1))/(yPos(i1)-yPos(i1-1));
    else
        xp = degreesRates(i1);
    end
    
    legend = text(x, y,[colName{i},': ',result.method,' snr=',num2str(result.snr),' dB, L=',int2str(result.L)]);
    set(legend,'BackgroundColor',[1,1,1]);
    plot([xp,x-0.5], [y,y], [colChar(i),'.-']);
    y = y+yPadding;
end

% print plot
legend = text(x, y,'Dictionary learning methods: ');
set(legend,'BackgroundColor',[1,1,1]);
print( gcf, '-depsc2', epsName );
disp('Printed figure as: .eps');
print( gcf, '-dpng', '-r80', pngName );
disp('Printed figure as: .png');

return