clc;clear all;close all;

addpath('eigen_analysis/');
addpath('mos/');
dataPath = '/media/thiago/shared/backup/doutorado/data/';

matrices = {'all';'signal';'noise';'portscan';'synflood';'fraggle'};
numberOfMatrices = size(matrices,1);                                        %6 matrices
ports = [80,443,53,21,22,23,25,110,143,161,69,123,445,600,19,67,68];
numberOfPorts = size(ports,2);                                              %17 ports
numPeriods = 6;
periodsSize = 20;
numMinutes = 120;
threshold = 0.4;

% incremental individualized
for matrix = 4:numberOfMatrices    
    switch matrix		
        case 1 %all
            q = 4;
        case 2 %signal			
            q = 4;
		case 3 %noise            
			q = 4;
		case 4 %portscan
            q = 3;
		case 5 %synflood 
            q = 4;
   		case 6 %fraggle 
			q = 5;
    end    
    X0 = dlmread([dataPath 'all/traffic/' num2str(2) '.txt'], '\t');
    if (q == 3)
        [S0,E0,Vr0,Mr0] = eigencorrelation(X0);
    else
        [S0,E0,Vr0,Mr0] = eigencovariance(X0);
    end
    X = dlmread([dataPath 'all/traffic/' num2str(q) '.txt'], '\t');
    t_attacks = {};
    for t = 1:periodsSize
        Xc = cat(2,X0,X(:,1:t));
        if (q == 3)
            [Sc,Ec,Vrc,Mrc] = eigencorrelation(Xc);
        else
            [Sc,Ec,Vrc,Mrc] = eigencovariance(Xc);
        end
        cosTheta(t) = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));
        cosTheta(t)  = abs(cosTheta(t));
        warning('Matrix = %s, Period(q) = %s, Time(t) = %s, cosTheta= %s',matrices{matrix},num2str(q),num2str(t),cosTheta(t));
        if(cosTheta(t) < threshold)
            X0 = cat(2,X0,X(:,1:t-1));
            ports = getportattack(X0,X(:,t),q == 3,threshold);
            %warning('Matrix = %s, Period(q) = %s, Time(t) = %s, Ports= %s',matrices{matrix},num2str(q),num2str(t),mat2str(ports));
            t = t + 1;                
            for a = t:periodsSize
                if (q == 3)
                    [S,E,Vrc,Mrc] = eigencorrelation(cat(2,X0,X(:,a)));
                else
                    [S,E,Vrc,Mrc] = eigencovariance(cat(2,X0,X(:,a)));
                end
                cosTheta(a) = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));                
                cosTheta(a)  = abs(cosTheta(a));
                warning('Matrix = %s, Period(q) = %s, Time(t) = %s, cosTheta= %s',matrices{matrix},num2str(q),num2str(a),cosTheta(t));
                if(cosTheta(a) < threshold)
                    ports = getportattack(X0,X(:,a),q == 3,threshold);
                    %warning('Matrix = %s, Period(q) = %s, Time(t) = %s, Ports= %s',matrices{matrix},num2str(q),num2str(a),mat2str(ports));
                end
            end
            break;
        end
    end        
    similarity{matrix} = cosTheta;
end

% incremental individualized
for matrix = 4:numberOfMatrices    
    switch matrix		
        case 1 %all
            q = 4;
        case 2 %signal			
            q = 4;
		case 3 %noise            
			q = 4;
		case 4 %portscan
            q = 3;
		case 5 %synflood 
            q = 4;
   		case 6 %fraggle 
			q = 5;
    end    
    X0 = dlmread([dataPath 'all/traffic/' num2str(2) '.txt'], '\t');
    if (q == 3)
        [S0,E0,Vr0,Mr0] = eigencorrelation(X0);
    else
        [S0,E0,Vr0,Mr0] = eigencovariance(X0);
    end
    X = dlmread([dataPath '/all/traffic/' num2str(q) '.txt'], '\t');
    t_attacks = {};
    for t = 1:periodsSize
        Xc = cat(2,X0,X(:,1:t));
        if (q == 3)
            [Sc,Ec,Vrc,Mrc] = eigencorrelation(Xc);
        else
            [Sc,Ec,Vrc,Mrc] = eigencovariance(Xc);
        end
        cosTheta(t) = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));
        cosTheta(t)  = abs(cosTheta(t));
        warning('Matrix = %s, Period(q) = %s, Time(t) = %s, cosTheta= %s',matrices{matrix},num2str(q),num2str(t),cosTheta(t));        
    end        
    similarity{matrix} = cosTheta;
end

% individual
for matrix = 4:numberOfMatrices    
    switch matrix		
        case 1 %all
            q = 4;
        case 2 %signal			
            q = 4;
		case 3 %noise            
			q = 4;
		case 4 %portscan
            q = 3;
		case 5 %synflood 
            q = 4;
   		case 6 %fraggle 
			q = 5;
    end    
    X0 = dlmread([dataPath 'all/traffic/' num2str(2) '.txt'], '\t');
    if (q == 3)
        [S0,E0,Vr0,Mr0] = eigencorrelation(X0);
    else
        [S0,E0,Vr0,Mr0] = eigencovariance(X0);
    end
    X = dlmread([dataPath '/all/traffic/' num2str(q) '.txt'], '\t');
    t_attacks = {};
    for t = 1:periodsSize
        Xc = cat(2,X0,X(:,t));  % only the target column
        if (q == 3)
            [Sc,Ec,Vrc,Mrc] = eigencorrelation(Xc);
        else
            [Sc,Ec,Vrc,Mrc] = eigencovariance(Xc);
        end
        cosTheta(t) = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));
        cosTheta(t)  = abs(cosTheta(t));
        warning('Matrix = %s, Period(q) = %s, Time(t) = %s, cosTheta= %s',matrices{matrix},num2str(q),num2str(t),cosTheta(t));        
    end        
    similarity{matrix} = cosTheta;
end