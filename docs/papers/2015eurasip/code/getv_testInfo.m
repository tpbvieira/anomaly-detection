function getv_testInfo()
    %matrix metadata
    matrices = {'all';'signal';'noise';'portscan';'synflood';'fraggle'};
    numberOfMatrices = size(matrices,1); %6 matrices
    ports = [80,443,53,21,22,23,25,110,143,161,69,123,445,600,19,67,68];
    numberOfPorts = size(ports,2); %17 ports    
    %minutes
    numPeriods = 6;
    periodsSize = 20;
    numMinutes = 120;    
    %seconds
    numSecPeriods = 120;
    periodsSecSize = 60;
    numSeconds = 7200;
    
    for matrix = 1:numberOfMatrices        
        %% Testing extracted values
        portMinutes = zeros(numberOfPorts,numMinutes);
        portSeconds = zeros(numberOfPorts,numMinutes * 60);
        mapMinutes = containers.Map;
        mapSeconds = containers.Map;
        [portMinutes, mapMinutes] = getv_getTimeFromFile(['data/' matrices{matrix} '/traffic/minutes_'], numberOfPorts, ports, 'minutes', portMinutes, mapMinutes);
        [portSeconds, mapSeconds] = getv_getTimeFromFile(['data/' matrices{matrix} '/traffic/seconds_'], numberOfPorts, ports, 'seconds', portSeconds, mapSeconds);
        if isequal(sum(sum(portMinutes)),sum(sum(portSeconds))) == 0
            warning('Error! %s has different ocurrencies for CALCULATED minutes (%s) and seconds (%s)',matrices{matrix},num2str(sum(sum(portMinutes))),num2str(sum(sum(portSeconds))));
        end

        %% Testing saved port x times
        portMinutes = 0;
        portSeconds = 0;
        for period = 1:numPeriods
            X = dlmread(['data/' matrices{matrix} '/traffic/' num2str(period) '.txt'], '\t');
            portMinutes = portMinutes + sum(sum(X));
        end
        
        for period = 1:numSecPeriods
            Xss = struct2cell(load(['data/' matrices{matrix} '/qs' num2str(period)]));
            portSeconds = portSeconds + sum(sum(Xss{1}));
        end
        
        if isequal(portMinutes,portSeconds) == 0
            warning('Error! %s saved different values for SAVED minutes (%s) and seconds (%s)',matrices{matrix},num2str(portMinutes),num2str(portSeconds));
        end
    end