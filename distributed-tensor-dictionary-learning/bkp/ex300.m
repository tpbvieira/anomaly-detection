% ex300           Test of sparseapprox.m, using AR(1) dictionary and data 
%
% example:
% clear all; ex300;  
% clear all; L = 400; doSlowMethods = true; ex300;  

%----------------------------------------------------------------------
% Copyright (c) 2011.  Karl Skretting.  All rights reserved.
% University of Stavanger, Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  06.04.2011  KS: test started
% Ver. 1.1  15.04.2013  KS: may set L before m-file is done
%----------------------------------------------------------------------

load('bestDforAR1.mat');   % load Dbest
load('dataXforAR1.mat');   % load X 
if ~exist('doSlowMethods', 'var')
    doSlowMethods = false;
end

s = 4;  
N = size(X,1);
tekst = char(' ', ['ex300 : results generated ',datestr(now)]);
if exist('L', 'var') && (L < size(X,2))
    I = randperm(size(X,2));
    X = X(:,I(1:L));  
    tekst = char(tekst, ['Use ',int2str(L),' of ',int2str(size(X,2)), ...
        ' randomly selected vectors from X.']);
end
tekst = char(tekst, 'OP is Orthogonal Projection (onto column space), MP is Matching Pursuit.');
tekst = char(tekst, 'w0, w1, r1, r2, and r2/x2 are mean values for 0, 1 or 2-norms ');
tekst = char(tekst, 'of x (signal), r (error) or w (coefficients).');
tekst = char(tekst, 'Summary: ');
K = size(Dbest,2);
L = size(X,2);

format1 = ' %30s :   time   SNR    w0     w1     r1     r2    r2/x2 ';
format2 = ' %30s :  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.3f ';
% 
tekst = char(tekst, sprintf(format1, 'Method'));
    
[W, res] = sparseapprox(X, Dbest, 'pinv', 'tnz',s, 'v',2, 'doOP',0);
t = sprintf(format2, 'pinv + threshold, not OP', ...
    res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
    mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
tekst = char(tekst, t);

[W, res] = sparseapprox(X, Dbest, 'pinv', 'tnz',s, 'v',2, 'doOP',1);
t = sprintf(format2, 'pinv + threshold and OP', ...
    res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
    mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
tekst = char(tekst, t);

[W, res] = sparseapprox(X, Dbest, 'mexLasso', 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'mexLasso in SPAMS', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[W, res] = sparseapprox(X, Dbest, 'mexLasso', 'tnz',6);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'mexLasso (s=6) in SPAMS', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

if doSlowMethods  % the SLOW methods linprog and FOCUSS
    [W, res] = sparseapprox(X, Dbest, 'linprog', 'tnz',s, 'doOP',false);
    t = sprintf(format2, 'linprog + threshold, not OP', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
    
    [W, res] = sparseapprox(X, Dbest, 'linprog', 'tnz',s, 'doOP',true);
    t = sprintf(format2, 'linprog + threshold and OP', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
    
    [W, res] = sparseapprox(X, Dbest, 'FOCUSS', ...
        'tnz',s, 'v',2, 'p', 0.8, 'l', 0.4, 'nIt', 100, 'doOP',false);
    t = sprintf(format2, 'FOCUSS + threshold, not OP', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
    
    [W, res] = sparseapprox(X, Dbest, 'FOCUSS', ...
        'tnz',s, 'v',2, 'p', 0.8, 'l', 0.4, 'nIt', 100, 'doOP',true);
    t = sprintf(format2, 'FOCUSS + threshold and OP', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[W, res] = sparseapprox(X, Dbest, 'javaBMP', 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'java Basic MP', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[W, res] = sparseapprox(X, Dbest, 'OMP', 'tnz',s, 'v',2);
t = sprintf(format2, 'OMP (matlab)', ...
    res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
    mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
tekst = char(tekst, t);

[W, res] = sparseapprox(X, Dbest, 'javaOMP', 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'javaOMP (mpv2)', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[W, res] = sparseapprox(X, Dbest, 'mexOMP', 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'mexOMP in SPAMS', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[W, res] = sparseapprox(X, Dbest, 'ORMP', 'tnz',s, 'v',2);
t = sprintf(format2, 'ORMP (matlab)', ...
    res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
    mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
tekst = char(tekst, t);

[W, res] = sparseapprox(X, Dbest, 'javaORMP', 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'javaORMP (mpv2)', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[W, res] = sparseapprox(X, Dbest, 'javaPS', 'nComb',10, 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'Partial Search (10)', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end
    
[W, res] = sparseapprox(X, Dbest, 'javaPS', 'nComb',250, 'tnz',s, 'v',2);
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'Partial Search (250)', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

[Wgmp, res] = sparseapprox(X, Dbest, 'GMP', 'tnz',L*s);
t = sprintf(format2, 'Global MP', ...
    res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
    mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
tekst = char(tekst, t);

[W, res] = sparseapprox(X, Dbest, 'javaPS', 'nComb',10, 'tnz',sum(Wgmp~=0));
if isfield(res,'Error')
    tekst = char(' ', res.Error, tekst);
else
    t = sprintf(format2, 'GMP + PS(10)', ...
        res.time, res.snr, mean(res.norm0W), mean(res.norm1W), mean(res.norm1R), ...
        mean(res.norm2R), mean(res.norm2R)/mean(res.norm2X) );
    tekst = char(tekst, t);
end

disp(tekst);