function y = eigensim(X,w,l,cov,mos)
% eigensim 
%
% SYNOPSIS: 
%
% INPUT X: data matrix with columns as observations and lines as variables
%       w: window size
%       l: number of windows for attack detection
%       cov: normalization method for covariance matrix. 'unit' for zero
%       mean and unit variance or 'zmean' for normalization by zero mean
%       mos: model order selection scheme. accepted models: 'akaike',
%       'edc', 'eft', 'radoi', 'sure'
%
% OUTPUT y: 
%
% EXAMPLE eigensim(X,w,l,norm,mos)
%
% SEE ALSO 
%
% created with MATLAB R2016a on Ubuntu 16.04
% created by: Thiago Vieira
% DATE: 01-Oct-2010
%

addpath('eigen_analysis/');
addpath('mos/');
addpath('util/');

nVar = size(X,1);
nObs = size(X,2);
numPeriods = l;
periodsSize = w;
threshold = 0.40;

% eigencovariance
i = 1;
maxEig = [];
for li = 1:numPeriods
    i2 = i+periodsSize-1; 
    X0 = X(:, i:i2);
	if(strcmp(cov,'unit'))
        [Sx,Ex,Vx,Mx] = eigencorrelation(X0);
    elseif(strcmp(cov,'zmean'))
        [Sx,Ex,Vx,Mx] = eigencovariance(X0);
    end
    i = i+periodsSize;
    maxEig(li) = Mx(1);
end

% MOS
numPCs = 0;
switch mos		
    case 'akaike'
        numPCs = akaike_short2(maxEig,periodsSize);
    case 'edc'
        numPCs = edc_short2(maxEig,periodsSize);
    case 'eft'
        c = struct2cell(load([dataPath 'Pfa']));
        Pfa = c{1};
        c = struct2cell(load([dataPath 'coeff']));
        coeff = c{1};
        c = struct2cell(load([dataPath 'q']));
        waiting = c{1};
        getv_size = size(maxEig);
        [eft_coeff,prob_found] = calc_coef_paper(getv_size(2),periodsSize,Pfa,coeff,waiting);        
        numPCs = eft_short(maxEig,eft_coeff,getv_size(2),periodsSize);
    case 'radoi'
        numPCs = ranoi_app(maxEig);
    case 'mdl'
        numPCs = mdl_short2(maxEig,periodsSize);
    case 'sure'
        numPCs = sure_method(sureM,nVar,periodsSize);
end

% gets index of periods pcs_q that are under attack
pcs_q = NaN;
maxEig_sorted = sort(maxEig,'descend');
for i = 1:numPCs
    pcs_q(i) = find(maxEig == maxEig_sorted(i));
end

% for each period under attack, gets the times t with attack
q_attacks = {};
for pc = 1:numPCs
    iRef = min(pcs_q) - 1;
    i2 = iRef * periodsSize;
    i1 = i2 - periodsSize + 1;
    X0 = X(:, i1:i2);                                                        % the reference traffic without attack
	if(strcmp(cov,'unit'))
        [S0,E0,Vr0,Mr0] = eigencorrelation(X0);
    elseif(strcmp(cov,'zmean'))
        [S0,E0,Vr0,Mr0] = eigencovariance(X0);
    end
    
    i2 = pcs_q(pc) * periodsSize;
    i1 = i2 - periodsSize + 1;
    X1 = X(:, i1:i2);
    
    t_attacks = {};
    
    for t = 1:periodsSize
        Xc = cat(2,X0,X1(:,1:t));
        if (strcmp(cov,'unit'))
            [Sc,Ec,Vrc,Mrc] = eigencorrelation(Xc);
        else
            [Sc,Ec,Vrc,Mrc] = eigencovariance(Xc);
        end
        cosTheta = dot(Vr0(:,Mr0(2)), Vrc(:,Mrc(2))) / (norm(Vr0(:,Mr0(2))) * norm(Vrc(:,Mrc(2))));
        cosTheta  = abs(cosTheta);
        warning('Period(q) = %s, Time(t) = %s, cosTheta= %s',num2str(pcs_q(pc)),num2str(t),cosTheta);
        if(cosTheta < threshold)
            t
            X0 = cat(2,X0,X(:,1:t-1));
            t_attacks = [t_attacks,t];                
            t = t + 1;                
            for a = t:periodsSize
                if (strcmp(cov,'unit'))
                    [S,E,Vrc,Mrc] = eigencorrelation(cat(2,X0,X(:,a)));
                else
                    [S,E,Vrc,Mrc] = eigencovariance(cat(2,X0,X(:,a)));
                end
                cosTheta = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));
                cosTheta  = abs(cosTheta);
                warning('Period(q) = %s, Time(t) = %s, cosTheta= %s',num2str(pcs_q(pc)),num2str(a),cosTheta);                    
                if(cosTheta < threshold)
                    a
                    t_attacks = [t_attacks,a];
                end
            end
            break;
        end
    end    
    q_attacks{pcs_q(pc)} = t_attacks
end