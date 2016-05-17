function P = mosEigenSimilarity_findport(X0,Xt,isCorr,threshold)
    P = [];
    numPorts = 0;
    T0 = X0(:,size(X0,2));        % last column of X0
    for p = 1:size(T0,1)
        T = T0;
        T(1:p) = Xt(1:p);           % concatenates until the first attack
        if (isCorr)
            [Sx,Ex,Vx,Mx] = eigencorrelation(X0);
            [Sp,Ep,Vp,Mp] = eigencorrelation(cat(2,X0,T));            
        else
            [Sx,Ex,Vx,Mx] = eigencovariance(X0);
            [Sp,Ep,Vp,Mp] = eigencovariance(cat(2,X0,T));
        end
        cosTheta = dot(Vx(:,Mx(2)),Vp(:,Mp(2)))/(norm(Vx(:,Mx(2)))*norm(Vp(:,Mp(2))));        
        cosTheta  = abs(cosTheta)
        if(cosTheta < threshold)
            numPorts = numPorts + 1;
            P(numPorts) = p;   
            T = T0;
            T(1:p-1) = Xt(1:p-1);      % copy ports without attack and use it as comparison
            p = p + 1;
            for pa = p:size(T0,1)
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