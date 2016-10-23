%% Extracts GETV Information
% It extracts traffic information from provided file and calculates GETV for kinds of traffic
% Extracts GETV information by different granularities 
function mosEigenSimilarity_generateData()

    addpath('/home/thiago/Dropbox/dev/projects/anomaly-detector/fontes/mos-eigen-similarity/src/main/matlab/eigen_analysis/');

    dataPath = '/media/thiago/shared/backup/doutorado/data/';
    resultPath = '/home/thiago/Dropbox/dev/projects/anomaly-detector/docs/papers/2016JNCA/results/';
    
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
    
    %time mapping
    mapMatrixMinutes = containers.Map;
    mapMatrixSeconds = containers.Map;

    for matrix = 1:numberOfMatrices
        portMinutes = zeros(numberOfPorts,numMinutes);
        portSeconds = zeros(numberOfPorts,numMinutes * 60);
        mapMinutes = containers.Map;
        mapSeconds = containers.Map;
        
        %% Data extraction
        % Creates matrix portMinutes, computing the number of packet per port x minute
        [portMinutes, mapMinutes] = mosEigenSimilarity_getTimeFromFile([dataPath matrices{matrix} '/traffic/minutes_'], numberOfPorts, ports, 'minutes', portMinutes, mapMinutes);
        %barplotfromlines(portMinutes, 'results/figures/', [matrices{matrix} 'PacketsPortMinutesBar'], 'png');
        %histplotfromlines(portMinutes, 'results/figures/', [matrices{matrix} 'PacketsPortMinutesHist'], 'png');

        %matrices{matrix}
        %'Minutes'
        %[D PD] = allfitdist(sum(portMinutes.'),'PDF'); %Compute and plot results 
        %D(1)
        
        % Creates matrix X, computing the number of packet per port x second
        [portSeconds, mapSeconds] = mosEigenSimilarity_getTimeFromFile([dataPath matrices{matrix} '/traffic/seconds_'], numberOfPorts, ports, 'seconds', portSeconds, mapSeconds);
        %barplotfromlines(portSeconds, 'results/figures/', [matrices{matrix} 'PacketsPortSecondsBar'], 'png');
        %histplotfromlines(portSeconds, 'results/figures/', [matrices{matrix} 'PacketsPortSecondsHist'], 'png');
        
        %matrices{matrix}
        %'Seconds'
        %[D PD] = allfitdist(sum(portSeconds.'),'PDF'); %Compute and plot results 
        %D(1)

        % Map with relationship between calculated  time and its readable value
        mapMatrixMinutes(num2str(matrix)) = mapMinutes;
        mapMatrixSeconds(num2str(matrix)) = mapSeconds;

        %% data splitting (minutes)
        % Divides a matrix portMinutes into q periods.
        % GETV shall analyze all traffic, without division by kind of
        % traffic, but it was adapted to result validation and to get the same results obtained by Danilo.
        %j = 0;
        for period = 1:numPeriods            
            switch matrix
                case 1 % all
%                     Xq = portMinutes(:,j-19:j);
%                     X(:,:,period) =  Xq;
%                     save([dataPath matrices{matrix} '/q' num2str(period)],'Xq');
                    %% Testing if values match
                    % testing if period match
                    X(:,:,period) = dlmread([dataPath matrices{matrix} '/traffic/' num2str(period) '.txt'], '\t');
%                     for port = 1:numberOfPorts
%                         for minute = 1:periodsSize
%                             if Xq(port,minute) ~= X(port,minute,period)
%                                 warning(['Matrix=' matrices{matrix} ', q=' num2str(period) ', port=' num2str(port) ', t=' num2str(minute) '. (' num2str(Xq(port,minute)) ' ~= ' num2str(X(port,minute,period)) ')']);
%                             end
%                         end
%                     end
                case 2 %signal
%                     Xq = portMinutes(:,j-19:j);
%                     X(:,:,period) =  Xq;
%                     save([dataPath matrices{matrix} '/q' num2str(period)],'Xq');
                case 3 %noise
%                     Xq = portMinutes(:,j-19:j);
%                     X(:,:,period) =  Xq;
%                     save([dataPath matrices{matrix} '/q' num2str(period)],'Xq');
                case 4 %portscan 
%                     Xq = portMinutes(:,j-19:j);
                    Signal = dlmread([dataPath matrices{2} '/traffic/' num2str(period) '.txt'], '\t');
                    Noise = dlmread([dataPath matrices{3} '/traffic/' num2str(period) '.txt'], '\t');                    
                    All = dlmread([dataPath matrices{1} '/traffic/' num2str(period) '.txt'], '\t');                    
                    X(:,:,period) = Signal + Noise + All;
                    dlmwrite([dataPath matrices{matrix} '/traffic/' num2str(period) '.txt'], X(:,:,period), '\t');
%                     for port = 1:numberOfPorts
%                         for minute = 1:periodsSize
%                             if Xq(port,minute) ~= X(port,minute,period)
%                                 warning(['Matrix=' matrices{matrix} ', q=' num2str(period) ', port=' num2str(port) ', t=' num2str(minute) '. (' num2str(Xq(port,minute)) ' ~= ' num2str(X(port,minute,period)) ')']);
%                             end
%                         end
%                     end
                case 5 %synflood
%                     Xq = portMinutes(:,j-19:j);
                    Signal = dlmread([dataPath matrices{2} '/traffic/' num2str(period) '.txt'], '\t');
                    Noise = dlmread([dataPath matrices{3} '/traffic/' num2str(period) '.txt'], '\t');
                    All = dlmread([dataPath matrices{1} '/traffic/' num2str(period) '.txt'], '\t');
                    Synflood = zeros(size(All,1),size(All,2));
                    Synflood(14,:) = All(14,:);	% synflood
                    Synflood(4:12,:) = All(4:12,:);	% portscan
                    X(:,:,period) = Signal + Noise + Synflood;
                    dlmwrite([dataPath matrices{matrix} '/traffic/' num2str(period) '.txt'], X(:,:,period), '\t');
%                     for port = 1:numberOfPorts
%                         for minute = 1:periodsSize
%                             if Xq(port,minute) ~= X(port,minute,period)
%                                 warning(['Matrix=' matrices{matrix} ', q=' num2str(period) ', port=' num2str(port) ', t=' num2str(minute) '. (' num2str(Xq(port,minute)) ' ~= ' num2str(X(port,minute,period)) ')']);
%                             end
%                         end
%                     end
                case 6 %fraggle 
%                     Xq = portMinutes(:,j-19:j);
                    Signal = dlmread([dataPath matrices{2} '/traffic/' num2str(period) '.txt'], '\t');
                    Noise = dlmread([dataPath matrices{3} '/traffic/' num2str(period) '.txt'], '\t');
                    All = dlmread([dataPath matrices{1} '/traffic/' num2str(period) '.txt'], '\t');
                    Fraggle = zeros(size(All,1),size(All,2));
                    Fraggle(15,:) = All(15,:);
                    X(:,:,period) = Signal + Noise + Fraggle;
                    dlmwrite([dataPath matrices{matrix} '/traffic/' num2str(period) '.txt'], X(:,:,period), '\t');                
%                     for port = 1:numberOfPorts
%                         for minute = 1:periodsSize
%                             if Xq(port,minute) ~= X(port,minute,period)
%                                 warning(['Matrix=' matrices{matrix} ', q=' num2str(period) ', port=' num2str(port) ', t=' num2str(minute) '. (' num2str(Xq(port,minute)) ' ~= ' num2str(X(port,minute,period)) ')']);
%                             end
%                         end
%                     end
            end
            
            %% calculate covariance/correlation, eigenvalues, eigenvectors and getv            
%             warning('Matrix = %s, Period = %s, Total = %s',matrices{matrix},num2str(period),num2str(sum(sum(X(:,:,period)))));
            %[a,b,c] = svd(X(:,:,period))
            [S,Es,Vs,Ms] = eigencovariance(X(:,:,period));
            Sqxx{period} = S;
            Esqxx{period} = Es;
            Vsqxx{period} = Vs;
            GETVsqxx{period} = Ms(1);		
            [R,Er,Vr,Mr] = eigencorrelation(X(:,:,period));
            Rqxx{period} = R;
            Erqxx{period} = Er;
            Vrqxx{period} = Vr;
            GETVrqxx{period} = Mr(1);            
        end
        
        %% data splitting (seconds)
        % Divides a matrix portSeconds into qs periods.
        for period = 1:numSecPeriods
            iniPeriod = (period*periodsSecSize) - periodsSecSize + 1;
            endPeriod = (period*periodsSecSize);
            Xss = portSeconds(:,iniPeriod:endPeriod);                    
            save([dataPath matrices{matrix} '/qs' num2str(period)],'Xss');
                        
            %% calculate covariance/correlation, eigenvalues, eigenvectors and getv        
            %warning('Matrix = %s, Period = %s, Total = %s',matrices{matrix},num2str(period),num2str(sum(sum(Xss))));
            [S,Es,Vs,Ms] = eigencovariance(Xss);
            Sqxxss{period} = S;
            Esqxxss{period} = Es;
            Vsqxxss{period} = Vs;
            GETVsqxxss{period} = Ms(1);		
            [R,Er,Vr,Mr] = eigencorrelation(Xss);
            Rqxxss{period} = R;
            Erqxxss{period} = Er;
            Vrqxxss{period} = Vr;
            GETVrqxxss{period} = Mr(1);            
        end
        
        % data persistence (minutes)
        save([dataPath matrices{matrix} '/Sqxx'],'Sqxx');            % covariance
        save([dataPath matrices{matrix} '/Esqxx'],'Esqxx');          % eigenvalues
        save([dataPath matrices{matrix} '/Vsqxx'],'Vsqxx');          % eigenvectors
        GETVsqxx_ = cell2mat(GETVsqxx);                               
        save([dataPath matrices{matrix} '/GETVsqxx'],'GETVsqxx_');	% getv 
        save([dataPath matrices{matrix} '/Rqxx'],'Rqxx');            % correlation
        save([dataPath matrices{matrix} '/Erqxx'],'Erqxx');          % eigenvalues
        save([dataPath matrices{matrix} '/Vrqxx'],'Vrqxx');          % eigevectors	
        GETVrqxx_ = cell2mat(GETVrqxx);								        
        save([dataPath matrices{matrix} '/GETVrqxx'],'GETVrqxx_');	% getv
        
        % data persistence (seconds)
        save([dataPath matrices{matrix} '/Sqxxss'],'Sqxxss');            % covariance
        save([dataPath matrices{matrix} '/Esqxxss'],'Esqxxss');          % eigenvalues
        save([dataPath matrices{matrix} '/Vsqxxss'],'Vsqxxss');          % eigenvectors
        GETVsqxxss_ = cell2mat(GETVsqxxss);                               
        save([dataPath matrices{matrix} '/GETVsqxxss'],'GETVsqxxss_');	 % getv 
        save([dataPath matrices{matrix} '/Rqxxss'],'Rqxxss');            % correlation
        save([dataPath matrices{matrix} '/Erqxxss'],'Erqxxss');          % eigenvalues
        save([dataPath matrices{matrix} '/Vrqxxss'],'Vrqxxss');          % eigevectors	
        GETVrqxxss_ = cell2mat(GETVrqxxss);								        
        save([dataPath matrices{matrix} '/GETVrqxxss'],'GETVrqxxss_');	% getv        
    end
    
    % data persistence of maps for future tracking
    save([resultPath 'mapMatrixMinutes'],'mapMatrixMinutes');
    save([resultPath 'mapMatrixSeconds'],'mapMatrixSeconds');