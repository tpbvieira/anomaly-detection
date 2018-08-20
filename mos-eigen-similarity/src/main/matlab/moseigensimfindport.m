function P = moseigensimfindport(X0,Xt,isCorr,threshold)
% moseigensimfindport seeks for ports under attack according to the defined
% threshold and the consine similarity analysis of changes on the principal
% eigenvectors 
%
% SYNOPSIS: a=testhelp(b,c)
%
% INPUT X0: reference matrix to be used for change comparison
%       Xt: vector for testing attack identification
%       isCorr: 1 if is correlation correlation (zero mean and unit 
%       variance) or 0 if is covariance matrix
%       threshold: minimal angle to identify no attack. Angles less than
%       the threshold indicates that one attack was found
%
% OUTPUT P: index of ports under attack or empty if no attack was found
%
% EXAMPLE moseigensimfindport(X0,X(:,t),1,0.8)
%
% SEE ALSO 
%
% created with MATLAB R2016a on Ubuntu 16.04
% created by: Thiago Vieira
% DATE: 01-Oct-2010
%

P = [];
numPorts = 0;
T0 = X0(:,size(X0,2));                                                      % last column of X0
for indPort = 1:size(T0,1)                                                  % for each port of the last column of X
    
    T = T0;
    T(1:indPort) = Xt(1:indPort);                                           % concatenates until the current port
    
    if (isCorr)
        [Sx,Ex,Vx,Mx] = eigencorrelation(X0);
        [Sp,Ep,Vp,Mp] = eigencorrelation(cat(2,X0,T));            
    else
        [Sx,Ex,Vx,Mx] = eigencovariance(X0);
        [Sp,Ep,Vp,Mp] = eigencovariance(cat(2,X0,T));
    end
    
    cosTheta = dot(Vx(:,Mx(2)),Vp(:,Mp(2)))/(norm(Vx(:,Mx(2)))*norm(Vp(:,Mp(2))));        
    cosTheta = abs(cosTheta)
    
    if(cosTheta < threshold)                                                % attack found
        numPorts = numPorts + 1;
        P(numPorts) = indPort;   
        T = T0;
        T(1:indPort-1) = Xt(1:indPort-1);                                   % copy ports without attack and use it as comparison
        indPort = indPort + 1;
        for pa = indPort:size(T0,1)                                         % since the fist attack was found, lets evaluate the remain ports
            V0pa = T;
            V0pa(pa) = Xt(pa);
            if (isCorr)
                [Sx,Ex,Vx,Mx] = eigencorrelation(X0);
                [Sp,Ep,Vp,Mp] = eigencorrelation(cat(2,X0,V0pa));            
            else
                [Sx,Ex,Vx,Mx] = eigencovariance(X0);
                [Sp,Ep,Vp,Mp] = eigencovariance(cat(2,X0,V0pa));
            end
            cosTheta = dot(Vx(:,Mx(2)),Vp(:,Mp(2)))/(norm(Vx(:,Mx(2)))*norm(Vp(:,Mp(2))));        
            cosTheta  = abs(cosTheta)
            if(cosTheta < threshold)
                numPorts = numPorts + 1;
                P(numPorts) = pa;   
            end
        end
        break
    end
    
end