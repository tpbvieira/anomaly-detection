addpath('/home/thiago/dev/projects/discriminative-sensing/mos-eigen-similarity/src/main/matlab/');

%matrix metadata
dataPath = '/media/thiago/shared/backup/doutorado/data/';
matrices = {'all';'signal';'noise';'portscan';'synflood';'fraggle'};
numberOfMatrices = size(matrices,1);                                        %6 matrices
ports = [80,443,53,21,22,23,25,110,143,161,69,123,445,600,19,67,68];
numberOfPorts = size(ports,2);                                              %17 ports    
%minutes
numPeriods = 6;
periodsSize = 20;
numMinutes = 120;    
%seconds
numSecPeriods = 120;
periodsSecSize = 60;
numSeconds = 7200;

for matrix = 1:numberOfMatrices        
    %% mosEigenSimilarity_getTimeFromFile
    portMinutes = zeros(numberOfPorts,numMinutes);
    portSeconds = zeros(numberOfPorts,numMinutes * 60);
    mapMinutes = containers.Map;
    mapSeconds = containers.Map;
    
    [portMinutes, mapMinutes] = mosEigenSimilarity_getTimeFromFile([dataPath matrices{matrix} '/traffic/minutes_'], numberOfPorts, ports, 'minutes', portMinutes, mapMinutes);
    [portSeconds, mapSeconds] = mosEigenSimilarity_getTimeFromFile([dataPath matrices{matrix} '/traffic/seconds_'], numberOfPorts, ports, 'seconds', portSeconds, mapSeconds);
    
    assert(sum(sum(portMinutes)) == sum(sum(portSeconds)))
    
    %% Testing saved port x times
    portMinutes = 0;
    portSeconds = 0;
    for period = 1:numPeriods
        X = dlmread([dataPath matrices{matrix} '/traffic/' num2str(period) '.txt'], '\t');
        portMinutes = portMinutes + sum(sum(X));
    end

    for period = 1:numSecPeriods
        Xss = struct2cell(load([dataPath matrices{matrix} '/qs' num2str(period)]));
        portSeconds = portSeconds + sum(sum(Xss{1}));
    end
    
end