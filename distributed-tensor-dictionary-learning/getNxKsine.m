function D = getNxKsine(N, K, betamin, L)
% getNxKsine      Get a NxK dictionary (frame) with sine elements
%                 Each atom is a sampled sine function, where the sample
% time (step) and the phase (start position) vary in an LxL grid, thus
% L^2 candidate atoms are possible.
% A candidate atom is added to the dictionary if the angle to all vectors 
% already included is larger than betamin. The search stop when K vectors
% has been used, or all L^2 candidates are tried (in this case the
% dictionary is completed by adding zeros, try smaller angle or larger L)
%
% D = getNxKsine(N, K, betamin);  
% D = getNxKsine(N, K, betamin, L);  
% 
% Third argument, betamin, is minimal angle between vectors in degrees.
% Fourth argument may be used to indicate how many different cadidate 
% atoms, sampled sine functions, to try. Default is 250.
% see also: a special variant, get8x21sine, which includes a plot option
%
% examples:
% D = getNxKsine(8, 25, 40, 250); 
% D = getNxKsine(8, 25, 48); 
% D = getNxKsine(8, 16, 60, 300);

if (nargin < 3); 
    disp('getNxKsine: should have 3 or 4 input arguments, see help.');
    D = 0;
    return;
end
if (nargin < 4); L = 250; end;
D = zeros(N,K);
fp = zeros(2,K);   % fs, phase

% try many candidate atoms
k = 1;
D(:,k) = ones(N,1)*sqrt(1/N);
clim = cos(betamin*pi/180);
for th = (1:(L-1))*(pi/L)
    for phi = (1:(L-1))*(pi/L);
        dsine = d(N,th,phi);
        c = D(:,1:k)'*dsine;  % cos of angles
        if (max(abs(c)) < clim)
            k = k+1;
            D(:,k) = dsine;
            fp(:,k) = [th, phi]';
        end
        if (k>K); break; end;
    end
    if (k>=K); break; end;
end
    
% sort the vectors according to frequency
% [temp,I] = sort(fp(1,:));
% D = D(:,I);
for k=1:K
    if (D(1,k) < 0)
        D(:,k) = -D(:,k);
    end
end

return;


% sample the sine function with 8 points, start at phase phi
% and go forward theta at each sample.
% Make the result orthogonal to DC-atom (scaled ones) or return DC
function dsine = d(N,theta,phi)
dsine = sin(phi+theta*(0:(N-1))');
dd = dsine'*dsine;
if dd>(1/(N*N))   
    dc = ones(N,1)*sqrt(1/N);
    dsine = dsine/sqrt(dd);
    dsine = dsine-(dc'*dsine)*dc;
    dd = dsine'*dsine;
    dsine = dsine/sqrt(dd);
else
    dsine = ones(N,1)/sqrt(N);
end
return
