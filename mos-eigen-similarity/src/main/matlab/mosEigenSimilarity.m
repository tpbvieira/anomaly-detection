clc; clear all; close all;

addpath('util/');
addpath('eigen_analysis/');
addpath('mos/');
dataPath = '/media/thiago/shared/backup/doutorado/data/';
resultPath = '../../../results/';

%[1]=80     
%[2]=443    
%[3]=53     
%[4]=21     portscan
%[5]=22     portscan
%[6]=23     portscan
%[7]=25     portscan
%[8]=110    portscan
%[9]=143    portscan
%[10]=161   portscan
%[11]=69    portscan
%[12]=123   portscan
%[13]=445	portscan
%[14]=600	synflood
%[15]=19	fraggle
%[16]=67    
%[17]=68    

%portscan	t=15
%synfood    t=11:20
%fragle     t=11:20

%output_precision(4)
%warning ('off', 'textscan: no data read');
%warning ('off', 'division by zero');
matrices = {'all';'signal';'noise';'portscan';'synflood';'fraggle'};
numberOfMatrices = size(matrices,1);                                        %6 matrices
ports = [80,443,53,21,22,23,25,110,143,161,69,123,445,600,19,67,68];
numberOfPorts = size(ports,2);                                              %17 ports
numPeriods = 6;
periodsSize = 20;
numMinutes = 120;
threshold = 0.40;

%% Extracts GETV Information
% extracts traffic information and calculates EVD for kinds of traffic
mosEigenSimilarity_evd(dataPath, resultPath);


%% eigen analysis
for matrix = 1:numPeriods    
    
    %portscan 
	if matrix == 4
		Erqxx_plot = zeros(5,numPeriods);
		periods = zeros(numPeriods,1);
		for period = 1:numPeriods
			periods(period) = period;            
            Erqxx = struct2cell(load([dataPath matrices{matrix} '/Erqxx']));
			Erq = diag(Erqxx{1}{period});
			for i = 1:size(Erq)
				if i > 5
					break;
              end
				Erqxx_plot(i,period) = Erq(i,1);
            end
        end
% 		plot(Erqxx_plot,'linestyle','-','marker','x');
%         set(gca,'FontSize',16);
% 		title('Portscan Eigenvalues');
% 		xlabel('Decrescent Largest Eigenvalues');
% 		ylabel('Eigenvalues');
% 		legend(strcat('q=',strtrim(cellstr(num2str(periods)))));
% 		print -depsc '/home/thiago/Dropbox/dev/projects/anomaly-detector/docs/papers/2016JNCA/results/figures/eigenvalues_portscan.eps'];
		%waitforbuttonpress();
    end
    
    %synflood or fraggle
	if matrix == 5 || matrix == 6
		Esqxx_plot = zeros(5,numPeriods);
		periods = zeros(numPeriods,1);
		for period = 1:numPeriods
			periods(period) = period;
            Esqxx = struct2cell(load([dataPath matrices{matrix} '/Esqxx']));
			Esq = diag(Esqxx{1}{period});
			for i = 1:size(Esq)
				if i > 5
					break;
                end
				Esqxx_plot(i,period) = Esq(i,1);
            end
        end
		if matrix == 5
			strTitle = 'Synflood Eigenvalues';
		elseif matrix == 6
			strTitle = 'Fraggle Eigenvalues';
        end
% 		plot(Esqxx_plot,'linestyle','-','marker','x');
%         set(gca,'FontSize',16);
% 		title(strTitle);
% 		xlabel('Decrescent Largest Eigenvalues');
% 		ylabel('Eigenvalues');
% 		legend(strcat('q=',strtrim(cellstr(num2str(periods)))));
% 		if matrix == 5
% 			print -depsc '/home/thiago/Dropbox/dev/projects/anomaly-detector/docs/papers/2016JNCA/results/figures/eigenvalues_synflood.eps';
% 		elseif matrix == 6
% 			print -depsc '/home/thiago/Dropbox/dev/projects/anomaly-detector/docs/papers/2016JNCA/results/figures/eigenvalues_fraggle.eps';
%         end
    end
end


%% MOS
% Model order selection and estimation of number of attacks
for matrix = 1:numberOfMatrices
    getv = NaN;
    Sqxx = NaN;
    sureM = NaN;
	switch matrix		
        case 1 %all
            getv = [1887545 2341327 3213867 133238294 92384021611 708335]; % covariance
            %getv = [2.0734 2.1451 10.0718 2.1620 2.4253 1.7948]; % correlation
            Sqxx = struct2cell(load([dataPath 'all/Rqxx']));
            sureM = Sqxx{1}{1};
        case 2 %signal			
            getv = [1887545 2341327 3213867 731229 6367983 708335]; % covariance
            %getv = [2.0734 2.1451 1.1930 2.1620 2.4253 1.7948]; % correlation
            Sqxx = struct2cell(load([dataPath 'signal/Rqxx']));
            sureM = Sqxx{1}{2};
		case 3 %noise            
			getv = [1887545 2341327 3213867 731229 6367983 708335]; % covariance
            %getv = [2.0734 2.1451 1.1930 2.1620 2.4253 1.7948]; % correlation
            Sqxx = struct2cell(load([dataPath 'noise/Rqxx']));
            sureM = Sqxx{1}{6};
		case 4 %portscan
            %getv = [1887545 2341327 3213867 731229 6367983 708335]; % covariance
            getv = [2.0734 2.1451 10.0718 2.1620 2.4253 1.7948]; % correlation
            Rqxx = struct2cell(load([dataPath 'portscan/Rqxx']));
            sureM = Rqxx{1}{3};
		case 5 %synflood 
            getv = [1887545 2341327 3213867 133238294 6367983 708335]; %covariance
            %getv = [2.0734 2.1451 1.1930 2.1620 2.4253 1.7948]; % correlation
            Sqxx = struct2cell(load([dataPath 'synflood/Rqxx']));
            sureM = Sqxx{1}{4};
   		case 6 %fraggle 
			getv = [1887545 2341327 3213867 731229 92384021611 708335]; % covariance
            %getv = [2.0734 2.1451 1.1930 2.1620 2.4253 1.7948]; % correlation
            Sqxx = struct2cell(load([dataPath 'fraggle/Rqxx']));
            sureM = Sqxx{1}{5};
    end
    
    t_mos = NaN(1,6);
    t_mos(1) = akaike_short2(getv,periodsSize); %AIC (1)
    t_mos(2) = mdl_short2(getv,periodsSize);    %MDL (2)    
    t_mos(3) = edc_short2(getv,periodsSize);    %EDC (3)
    t_mos(4) = ranoi_app(getv);                 %RADOI (4)
    c = struct2cell(load([dataPath 'Pfa']));
    Pfa = c{1};
    c = struct2cell(load([dataPath 'coeff']));
    coeff = c{1};
    c = struct2cell(load([dataPath 'q']));
    waiting = c{1};
    getv_size = size(getv);
    %[eft_coeff,prob_found] = calc_coef_paper(getv_size(2),periodsSize,Pfa,coeff,waiting);
    eft_coeff = [0.3000    0.4000    0.4000    0.4000    0.4000];
    t_mos(5) = eft_short(getv,eft_coeff,getv_size(2),periodsSize);      %EFT (5)
    t_mos(6) = sure_method(sureM,numberOfPorts,periodsSize);            %SURE (6)
    mos{matrix} = t_mos;

    %% gets what periods pcs_q are under attack
    pcs_q = NaN;
    numPCs = t_mos(3); 
    getv_sorted = sort(getv,'descend');    
    for pc = 1:numPCs
        pcs_q(pc) = find(getv == getv_sorted(pc)); % get the index of the numPCs largest getv
    end
    pcs{matrix} = pcs_q;
    
    %% for each period under attack, gets the times t and ports with attack
    q_attacks = {};
    for pc = 1:numPCs
        
        X0 = dlmread([dataPath matrices{matrix} '/traffic/' num2str(pcs_q(pc)-1) '.txt'], '\t');  % the reference traffic withtou attack
        if (pcs_q(pc) == 3)
            [S0,E0,Vr0,Mr0] = eigencorrelation(X0);
        else
            [S0,E0,Vr0,Mr0] = eigencovariance(X0);
        end
        X = dlmread([dataPath matrices{matrix} '/traffic/' num2str(pcs_q(pc)) '.txt'], '\t');     % traffi of the period q under attack
        t_attacks = {};
        
        for t = 1:periodsSize
            
            Xc = cat(2,X0,X(:,1:t));
            if (pcs_q(pc) == 3)
                [Sc,Ec,Vrc,Mrc] = eigencorrelation(Xc);
            else
                [Sc,Ec,Vrc,Mrc] = eigencovariance(Xc);
            end
            
            cosTheta = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2))) / (norm(Vr0(:,Mr0(2))) * norm(Vrc(:,Mrc(2))));            
            cosTheta  = abs(cosTheta);
            %warning('Matrix = %s, Period(q) = %s, Time(t) = %s, cosTheta= %s',matrices{matrix},num2str(pcs_q(pc)),num2str(t),cosTheta);
            if(cosTheta < threshold)
                X0 = cat(2,X0,X(:,1:t-1));
                t_attacks = [t_attacks,t];                
                ports = mosEigenSimilarity_findport(X0,X(:,t),pcs_q(pc) == 3,threshold);
                %warning('Matrix = %s, Period(q) = %s, Time(t) = %s, Ports= %s',matrices{matrix},num2str(pcs_q(pc)),num2str(t),mat2str(ports));
                t = t + 1;                
                for a = t:periodsSize
                    if (pcs_q(pc) == 3)
                        [S,E,Vrc,Mrc] = eigencorrelation(cat(2,X0,X(:,a)));
                    else
                        [S,E,Vrc,Mrc] = eigencovariance(cat(2,X0,X(:,a)));
                    end
                    cosTheta = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));
                    cosTheta  = abs(cosTheta);
                    warning('Matrix = %s, Period(q) = %s, Time(t) = %s, cosTheta= %s',matrices{matrix},num2str(pcs_q(pc)),num2str(a),cosTheta);                    
                    if(cosTheta < threshold)
                        t_attacks = [t_attacks,a];
                        ports = mosEigenSimilarity_findport(X0,X(:,a),pcs_q(pc) == 3,threshold);
                        %warning('Matrix = %s, Period(q) = %s, Time(t) = %s, Ports= %s',matrices{matrix},num2str(pcs_q(pc)),num2str(a),mat2str(ports));
                    end
                end
                break;
            end
        end        
        q_attacks{pcs_q(pc)} = t_attacks;
        attacks{matrix} = q_attacks;
    end
        
    %% for each time t, find attacked ports    
    for i = 1:size(q_attacks,2)
        if (size(q_attacks{i}) > 0)
            for ta = 1:size(q_attacks{i},2)
                q_attacks{i}{ta};
            end            
        end
    end
end