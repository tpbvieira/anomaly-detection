function y = eigensim(X,w,l,cov,mos, th)
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
periodSize = w;
threshold = th;
y = zeros(1,nObs);

% eigencovariance for each period to obtain the largest eigenvalues 
X0Ini = 1;
X0End = X0Ini + periodSize - 1; 
maxEig = [];
for li = 1:numPeriods
    X0 = X(:, X0Ini:X0End);
	if(strcmp(cov,'unit'))
        [Sx,Ex,Vx,Mx] = eigencorrelation(X0);
    elseif(strcmp(cov,'zmean'))
        [Sx,Ex,Vx,Mx] = eigencovariance(X0);
    end
    maxEig(li) = Mx(1);
    X0Ini = X0Ini + periodSize; 
    X0End = X0Ini + periodSize - 1; 
end

% MOS from largest eigenvalues to predic the number of attacks
nAttacks = 0;
switch mos		
    case 'akaike'
        nAttacks = akaike_short2(maxEig, periodSize);
    case 'edc'
        nAttacks = edc_short2(maxEig, periodSize);
    case 'eft'
        c = struct2cell(load([dataPath 'Pfa']));
        Pfa = c{1};
        c = struct2cell(load([dataPath 'coeff']));
        coeff = c{1};
        c = struct2cell(load([dataPath 'q']));
        waiting = c{1};
        getv_size = size(maxEig);
        [eft_coeff,prob_found] = calc_coef_paper(getv_size(2),periodSize,Pfa,coeff,waiting);        
        nAttacks = eft_short(maxEig,eft_coeff,getv_size(2),periodSize);
    case 'radoi'
        nAttacks = ranoi_app(maxEig);
    case 'mdl'
        nAttacks = mdl_short2(maxEig,periodSize);
    case 'sure'
        nAttacks = sure_method(sureM,nVar,periodSize);
end

% gets index of periods that are under attack
attacked_q = NaN;
maxEig_sorted = sort(maxEig,'descend');
for i = 1:nAttacks
    attacked_q(i) = find(maxEig == maxEig_sorted(i));
end

% for each period under attack, gets the times with attack
iInit = 1;
iEnd = periodSize;
qRef = min(attacked_q) - 1;
X0End = qRef * periodSize;
X0Ini = X0End - periodSize + 1;
X0 = X(:, X0Ini:X0End);                                                   % the reference traffic without attack
if(strcmp(cov,'unit'))
    [S0,E0,Vr0,Mr0] = eigencorrelation(X0);
elseif(strcmp(cov,'zmean'))
    [S0,E0,Vr0,Mr0] = eigencovariance(X0);
end
for iAtt = 1:nAttacks    
    X0End = attacked_q(iAtt) * periodSize;
    X0Ini = X0End - periodSize + 1;
    
    for t = X0Ini:X0End                                                   
        Xc = cat(2, X0, X(:,X0Ini:t));                                    % incremental
        if (strcmp(cov,'unit'))
            [Sc,Ec,Vrc,Mrc] = eigencorrelation(Xc);
        else
            [Sc,Ec,Vrc,Mrc] = eigencovariance(Xc);
        end
        cosTheta = dot(Vr0(:,Mr0(2)), Vrc(:,Mrc(2))) / (norm(Vr0(:,Mr0(2))) * norm(Vrc(:,Mrc(2))));
        cosTheta  = abs(cosTheta);
%         warning('Period(q) = %s, Time(t) = %s, cosTheta= %s',num2str(attacked_q(iAtt)),num2str(t),cosTheta);
        if(cosTheta < threshold)
            y(1,t) = 1;                                                       % the time t is under attack and 
            X0 = cat(2,X0, X(:,X0Ini:t-1));
            tIni = t + 1;
            for t = tIni:X0End                                              % individualized
                if (strcmp(cov,'unit'))
                    [S,E,Vrc,Mrc] = eigencorrelation(cat(2, X0, X(:,t)));
                else
                    [S,E,Vrc,Mrc] = eigencovariance(cat(2, X0, X(:,t)));
                end
                cosTheta = dot(Vr0(:,Mr0(2)),Vrc(:,Mrc(2)))/(norm(Vr0(:,Mr0(2)))*norm(Vrc(:,Mrc(2))));
                cosTheta  = abs(cosTheta);
%                 warning('Period(q) = %s, Time(t) = %s, cosTheta= %s',num2str(attacked_q(iAtt)),num2str(t),cosTheta);                    
                if(cosTheta < threshold)
                    y(1,t) = 1;                                               % the time t is under attack and 
                end
            end
            break;
        end
    end    
end