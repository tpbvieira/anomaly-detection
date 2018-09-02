clc; close all; clear;

% initiate variables
addpath('../../../main/matlab/rsvd/rsvd');
addpath('../../../main/matlab/rsvd/rSVD-single-pass');

A = randn(4000,100)*randn(100,1000);
fprintf('Matrix Size: %dx%d', size(A,1), size(A,2));

% full svd
fprintf('\n\n(4000x1000)')
tic; 
[U0,S0,V0] = svd(A);
time = toc;
fprintf('\nsvd:\tdif=%.4f,\tseconds=%.4f\n', norm(U0*S0*V0'-U0*S0*V0'), time);

% svd for 1 principal component
fprintf('\n(4000x1000,1)')
tic;
[U11,S11,V11] = rsvd(A,1);
time = toc;
fprintf('\nrsvd:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U11*S11*V11'), time);
tic;
[U13,S13,V13] = rSVDbasic(A,1);
time = toc;
fprintf('\nrSVDbasic:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U13*S13*V13'), time);
tic;
[U14,S14,V14] = rSVDsp(A,1);
time = toc;
fprintf('\nrSVDsp:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U14*S14*V14'), time);
tic;
[U12,S12,V12] = rSVD_exSP(A,1);
time = toc;
fprintf('\nrSVD_exSP:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U12*S12*V12'), time);

% svd for 50 principal components
fprintf('\n\n(4000x1000,50)')
tic;
[U21,S21,V21] = rsvd(A,50);
time = toc;
fprintf('\nrsvd:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U21*S21*V21'), time);
tic;
[U23,S23,V23] = rSVDbasic(A,50);
time = toc;
fprintf('\nrSVDbasic:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U23*diag(S23)*V23'), time);
tic;
[U24,S24,V24] = rSVDsp(A,50);
time = toc;
fprintf('\nrSVDsp:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U24*diag(S24)*V24'), time);
tic;
[U22,S22,V22] = rSVD_exSP(A,50);
time = toc;
fprintf('\nrSVD_exSP:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U22*diag(S22)*V22'), time);

% svd for 100 principal components
fprintf('\n\n(4000x1000,100)')
tic;
[U21,S21,V21] = rsvd(A,100);
time = toc;
fprintf('\nrsvd:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U21*S21*V21'), time);
tic;
[U23,S23,V23] = rSVDbasic(A,100);
time = toc;
fprintf('\nrSVDbasic:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U23*diag(S23)*V23'), time);
tic;
[U24,S24,V24] = rSVDsp(A,100);
time = toc;
fprintf('\nrSVDsp:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U24*diag(S24)*V24'), time);
tic;
[U22,S22,V22] = rSVD_exSP(A,100);
time = toc;
fprintf('\nrSVD_exSP:\tdif=%.4f,\tseconds=%.4f', norm(U0*S0*V0'-U22*diag(S22)*V22'), time);