clear; % clear variables
clc; % clear console
close all; % close figures

addpath('../../lib/matlab/LIBRA_20160628');

rng default
rho = [1,0.05;0.05,1];
u = copularnd('Gaussian',rho,50);
noise = randperm(50,5);
u(noise,1) = u(noise,1)*5;

[Sfmcd, Mfmcd, dfmcd, Outfmcd] = robustcov(u);
[Sogk, Mogk, dogk, Outogk] = robustcov(u,'Method','ogk');
[Soh, Moh, doh, Outoh] = robustcov(u,'Method','olivehawkins');
[rew,raw,hsetsfull] = DetMCD(u,'plots',0);

%%
% Calculate the classical distance values for the sample data using the
% Mahalanobis measure.
d_classical = pdist2(u, mean(u),'mahal');
p = size(u,2);
chi2quantile = sqrt(chi2inv(0.975,p));

%%
% Create DD Plots for each robust covariance calculation method.
figure
subplot(2,2,1)
plot(d_classical, dfmcd, 'o')
line([chi2quantile, chi2quantile], [0, 30], 'color', 'r')
line([0, 6], [chi2quantile, chi2quantile], 'color', 'r')
hold on
plot(d_classical(Outfmcd), dfmcd(Outfmcd), 'r+')
xlabel('Mahalanobis Distance')
ylabel('Robust Distance')
title('DD Plot, FMCD method')
hold off

subplot(2,2,2)
plot(d_classical, dogk, 'o')
line([chi2quantile, chi2quantile], [0, 30], 'color', 'r')
line([0, 6], [chi2quantile, chi2quantile], 'color', 'r')
hold on
plot(d_classical(Outogk), dogk(Outogk), 'r+')
xlabel('Mahalanobis Distance')
ylabel('Robust Distance')
title('DD Plot, OGK method')
hold off

subplot(2,2,3)
plot(d_classical, doh, 'o')
line([chi2quantile, chi2quantile], [0, 30], 'color', 'r')
line([0, 6], [chi2quantile, chi2quantile], 'color', 'r')
hold on
plot(d_classical(Outoh), doh(Outoh), 'r+')
xlabel('Mahalanobis Distance')
ylabel('Robust Distance')
title('DD Plot, Olive-Hawkins method')
hold off


%%
% In a DD plot, the data points tend to cluster in a straight line that
% passes through the origin. Points that are far removed from this line are
% generally considered outliers. In each of the previous plots, the red '+'
% symbol indicates the data points that |robustcov| considers to be
% outliers. 