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
clf;
clear all;
set(0,'RecursionLimit',1200)
scriptName = 'dictionary_learning';

%% parameters
s = 5;           % sparseness
snr = 20;        % snr for added noise
L = 500;         % number of training vectors to use
nofTrials = 3;   % enought trials to obtain reliable results
noIt = 100;      % number of iterations in each trial
N = 80;
K = 200;
M1 = 10;
M2 = 8;
N1 = 20;
N2 = 10;

%% select the methods to compare and define file names
fileNameInfo = sprintf('%1i_%li_%li_%li_%li',s,snr,L,N*K,noIt);
fileNameSufix = sprintf('%s.mat',fileNameInfo);
dataFiles = [
             ['L', fileNameSufix] % 'L', 'Q', 'C', 'H' or 'E' = RLS-DLA (java),
             ['T', fileNameSufix] % 'T' = K-HOSVD
             ['O', fileNameSufix] % 'O' = T-MOD,             
             ['K', fileNameSufix] % 'K' = K-SVD,   
             ['D', fileNameSufix] % 'D' = MOD,
             %['A', fileNameSufix] % 'A' = AK-SVD,
             %['M', fileNameSufix] % 'M' = ILS-DLA MOD,             
             %['I', fileNameSufix] % 'I' = ILS-DLA MOD (java),             
             %['B', fileNameSufix] % 'B' = RLS-DLA miniBatch             
             ];
numMethods = size(dataFiles,1);
methodNames = cell(numMethods,1);

%% plot configuration
epsName = sprintf('%1i_%li_%li_%li_%li.eps',s,snr,L,N*K,noIt);
%pngName = sprintf('%1i_%li_%li_%li_%li.png',s,snr,L,N*K,noIt);
colors = 'brgmyck';                                                         %'Blue', 'Red', 'Green', 'Magenta', 'Yellow', 'Cyan', 'Black'
degreesRates = [0.25:0.25:10, 10.5:0.5:25];
confidence = 0.1;                                                           % percentual of trials required to have confidence
betalim = 8.11;                                                             % Limiar: 1 - d'*dorg < 0.01  ==> |cos(beta)| > 0.01
hold on;
grid on;

%% for selected methods for comparison
for i=1:numMethods;  
    % load or make the data
    if exist(dataFiles(i,:),'file')
        fileName = dir(dataFiles(i,:));
        disp([scriptName,': results of ',dataFiles(i,:),' at ',fileName.date,'.']);
        load(dataFiles(i,:));                                               % try load privious results
        trialsDone = results.nofTrials;
        if results.noIt ~= noIt
            disp(['WARN : Number of iterations in file is ',int2str(results.noIt),...
                  ' while wanted iterations is ',int2str(noIt),' **.']);
        end
    else
        trialsDone = 0;
    end    
    
    % execute remain trials for atoms identification
    if (nofTrials > trialsDone)
        methodChar = dataFiles(i,1);        
        results = execDL(L, N, K, M1, M2, N1, N2, snr, methodChar, s, noIt, nofTrials-trialsDone, betalim);
    end
    methodNames{i} = results.method;
    
    % prepare cumulative atom identificatin per degree rates
    yPos = zeros(size(K,1),1);
    for i1=1:numel(degreesRates);
        yPos(i1) = sum(results.beta(:) < degreesRates(i1));
    end
    yPos = yPos/results.nofTrials;                                          % simple mean by trials
    
    % select identifications that happens in a percentual of trials
    I = find(yPos > (results.nofTrials * confidence));
    i1 = I(1);
    
    % update plot for the current method
    plot(degreesRates, yPos, colors(i));
end

%% print plot
title({'Number of dictionary atoms identified per degrees.'; 'Elements: ';N*K});
ylabel('Number of identified atoms.');
xlabel('Required degrees for identification.');
legend(methodNames, 'Location','SouthEast');
hold off;
print( gcf, '-depsc2', epsName );
disp('Printed figure as: .eps');
%print( gcf, '-dpng', '-r80', pngName );
%disp('Printed figure as: .png');

return