% bestAR1         Load the best dictionary for the AR(1) data into workspace.
%                 Also find and display SNR for this dictionary for
% the set of training vectors used. 
% This dictionary is only the best one found so far for these AR1 data.
% The quality is measured by vector selection using ORMP and s=4.
% Note workspace is cleared!
% 
% use:
%   bestAR1

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger, Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  29.04.2009  KS: test started
% Ver. 1.1  22.06.2009  KS: use in dle (Dictionary Learning Experiment)
%----------------------------------------------------------------------

clear all
load('bestDforAR1.mat');   % load Dbest
load('dataXforAR1.mat');   % load X 

s = 4;  
N = size(X,1);
L = size(X,2);
K = size(Dbest,2);

disp(' ');
disp('bestAR1: Start to find a sparse representation of the data.');
disp(['(s=',int2str(s),', N=',int2str(N),', K=',int2str(K),' and L=',int2str(L),')']);

W = sparseapprox(X, Dbest, 'javaORMP', 'tnz',s);
R = X - Dbest*W;

snr = 10*log10(sum(sum(X.*X))/sum(sum(R.*R)));
disp(['Achieved SNR using dictionary is ',num2str(snr),' dB.']);

% compare to DCT with thresholding
Y = dct(X);
for j=1:L
    [y,I] = sort(abs(Y(:,j)));
    Y(I(1:(N-s)),j) = 0;
end
R = X - idct(Y);
snr = 10*log10(sum(sum(X.*X))/sum(sum(R.*R)));
disp(['Achieved SNR using DCT and thresholding is ',num2str(snr),' dB.']);

% compare to KLT with thresholding
[T,d] = eig(X*X');
Y = T'*X;
for j=1:L
    [y,I] = sort(abs(Y(:,j)));
    Y(I(1:(N-s)),j) = 0;
end
R = X - T*Y;
snr = 10*log10(sum(sum(X.*X))/sum(sum(R.*R)));
disp(['Achieved SNR using KLT and thresholding is ',num2str(snr),' dB.']);
