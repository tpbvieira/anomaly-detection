clc; clear all; close all;

% data
X1 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/1.txt', '\t');
X2 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/2.txt', '\t');
X3 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/3.txt', '\t');
X4 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/4.txt', '\t');
X5 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/5.txt', '\t');
X6 = dlmread('/media/thiago/ubuntu/datasets/network/data/all/traffic/6.txt', '\t');

% OK 1 - SeasonOk, WindowOK. Middle Errors. Scan=3/55.DDoS=4/71-80,5/91-100
X1Test = [X1,X2,X3,X4,X5,X6];
y1TestUnit = zeros(1,120);
y1TestUnit(55) = 1;
y1TestZmean = zeros(1,120);
y1TestZmean(71:80) = 1;
y1TestZmean(91:100) = 1;

% OK 2 - SeasonOk, WindowOK. Early Errors. Scan=1/15.DDoS=4/71-80,5/91-100
X2Test = [X3,X2,X1,X4,X5,X6];
y2TestUnit = zeros(1,120);
y2TestUnit(11) = 1; % False positive
y2TestUnit(15) = 1;
y2TestZmean = zeros(1,120);
y2TestZmean(71:80) = 1;
y2TestZmean(91:100) = 1;

% OK 3 - SeasonOk, WindowOK. Early Errors. Scan=3/55.DDoS=1/11-20,4/71-80
X3Test = [X5,X2,X3,X4,X1,X6];
y3TestUnit = zeros(1,120);
y3TestUnit(55) = 1;
y3TestZmean = zeros(1,120);
y3TestZmean(11:20) = 1;
y3TestZmean(71:80) = 1;

% OK 4 - SeasonOk, WindowOK. Final Errors. Scan=3/55.DDoS=4/71-80,6/111-120
X4Test = [X1,X2,X3,X4,X6,X5];
y4TestUnit = zeros(1,120);
y4TestUnit(55) = 1;
y4TestZmean = zeros(1,120);
y4TestZmean(71:80) = 1;
y4TestZmean(111:120) = 1;

% OK 5 - SeasonOk, WindowOK. Final Errors. Scan=6/115.DDoS=4/71-80,3/51-60
X5Test = [X1,X2,X5,X4,X6,X3]; 
y5TestUnit = zeros(1,120);
y5TestUnit(115) = 1;
y5TestZmean = zeros(1,120);
y5TestZmean(71:80) = 1;
y5TestZmean(51:60) = 1;

% 6 - SeasonNOk, WindowNOK. Middle Errors. Scan=3/55.DDoS=4/71-80,5/91-100,
% 7/121-129
X6Test = [X1Test,X5(:,9:17),X1];
y6TestUnit = zeros(1,149);
y6TestUnit(55) = 1;
y6TestZmean = zeros(1,149);
y6TestZmean(71:80) = 1;
y6TestZmean(91:100) = 1;
y6TestZmean(123:129) = 1;

% 7 - SeasonNOk, WindowNOK. Early Errors.
% Scan=1/15,4/75.DDoS=5/91-100,6/111-120,7/121-129
X7Test = [X3,X1Test,X5(:,9:17),X1];
y7TestUnit = zeros(1,169);
y7TestUnit(15) = 1;
y7TestUnit(75) = 1;
y7TestZmean = zeros(1,169);
y7TestZmean(91:100) = 1;
y7TestZmean(111:120) = 1;
y7TestZmean(123:129) = 1;

% 8 - SeasonNOk, WindowNOK. Early Errors.
% Scan=4/75.DDoS=1/11/20,5/91-100,6/111-120,7/121-129 
X8Test = [X4,X1Test,X5(:,9:17),X1];
y8TestUnit = zeros(1,169);
y8TestUnit(75) = 1;
y8TestZmean = zeros(1,169);
y8TestZmean(11:20) = 1;
y8TestZmean(91:100) = 1;
y8TestZmean(111:120) = 1;
y8TestZmean(123:129) = 1;

% 9 - SeasonNOk, WindowNOK. Final Errors.
% Scan=3/15,8/144.DDoS=4/71-80,5/111-120,7/121-129
X9Test = [X1Test,X5(:,9:17),X3];
y9TestUnit = zeros(1,149);
y9TestUnit(55) = 1;
y9TestUnit(144) = 1;
y9TestZmean = zeros(1,140);
y9TestZmean(71:80) = 1;
y9TestZmean(91:100) = 1;
y9TestZmean(123:129) = 1;

% 10 - SeasonNOk, WindowNOK. Final Errors.
% Scan=3/55.DDoS=4/71-80,5/111-120,7/121-129,8/141-149
X10Test = [X1Test,X5(:,9:17),X5];
y10TestUnit = zeros(1,149);
y10TestUnit(55) = 1;
y10TestZmean = zeros(1,149);
y10TestZmean(71:80) = 1;
y10TestZmean(91:100) = 1;
y10TestZmean(123:129) = 1;
y10TestZmean(140:149) = 1;

% 11 - SeasonNOk, WindowOK. Middle Errors. Scan=3/15.DDoS=4/11-20,5/11-20
X11Test = [X1Test,X1];
y11TestUnit = zeros(1,140);
y11TestUnit(55) = 1;
y11TestZmean = zeros(1,140);
y11TestZmean(71:80) = 1;
y11TestZmean(91:100) = 1;

% 12 - SeasonNOk, WindowOK. Early Errors. Scan=1/15.DDoS=4/11-20,5/11-20
X12Test = [X3,X1Test]; 
y12TestUnit = zeros(1,140);
y12TestUnit(15) = 1;
y12TestUnit(75) = 1;
y12TestZmean = zeros(1,140);
y12TestZmean(91:100) = 1;
y12TestZmean(111:120) = 1;

% 13 - SeasonNOk, WindowOK. Early Errors. Scan=3/15.DDoS=1/11-20,4/11-20
X13Test = [X4,X1Test];
y13TestUnit = zeros(1,140);
y13TestUnit(75) = 1;
y13TestZmean = zeros(1,140);
y13TestZmean(11:20) = 1;
y13TestZmean(91:100) = 1;
y13TestZmean(111:120) = 1;

% 14 - SeasonNOk, WindowOK. Final Errors. Scan=3/55,7/135.DDoS=4/71-80,5/91-100
X14Test = [X1Test,X3];
y14TestUnit = zeros(1,140);
y14TestUnit(55) = 1;
y14TestZmean = zeros(1,140);
y14TestZmean(71:80) = 1;
y14TestZmean(91:100) = 1;

% 15 - SeasonNOk, WindowOK. Final Errors.
% Scan=3/55.DDoS=4/71-80,5/91-100,7/131-140
X15Test = [X1Test,X4];
y15TestUnit = zeros(1,140);
y15TestUnit(55) = 1;
y15TestZmean = zeros(1,140);
y15TestZmean(71:80) = 1;
y15TestZmean(91:100) = 1;

% %% Test Scenario 1 - unit
% y = eigensim(X1Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y1TestUnit))
% 
% %% Test Scenario 1 - zmean
% y = eigensim(X1Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y1TestZmean))
% 
% %% Test Scenario 2 - unit
% y = eigensim(X2Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y2TestUnit))
% 
% %% Test Scenario 2 - zmean
% y = eigensim(X2Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y2TestZmean))
% 
% %% Test Scenario 3 - unit
% y = eigensim(X3Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y3TestUnit))
% 
% %% Test Scenario 3 - zmean
% y = eigensim(X3Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y3TestZmean))
% 
% %% Test Scenario 4 - unit
% y = eigensim(X4Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y4TestUnit))
% 
% %% Test Scenario 4 - zmean
% y = eigensim(X4Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y4TestZmean))
% 
% %% Test Scenario 5 - unit
% y = eigensim(X5Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y5TestUnit))
% 
% %% Test Scenario 5 - zmean
% y = eigensim(X5Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y5TestZmean))
% 
% %% Test Scenario 6 - unit
% y = eigensim(X6Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y6TestUnit))
% 
% %% Test Scenario 6 - zmean
% y = eigensim(X6Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y6TestZmean))

%% Test Scenario 7 - unit
y = eigensim(X7Test,20,6,'unit','edc', 0.40);
y==y7TestUnit
assert(isequal(y,y7TestUnit))

%% Test Scenario 7 - zmean
y = eigensim(X7Test,20,6,'zmean','edc', 0.40);
assert(isequal(y,y7TestZmean))

% %% Test Scenario 8 - unit
% y = eigensim(X8Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y8TestUnit))
% 
% %% Test Scenario 8 - zmean
% y = eigensim(X8Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y8TestZmean))

% %% Test Scenario 9 - unit
% y = eigensim(X9Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y9TestUnit))
% 
% %% Test Scenario 9 - zmean
% y = eigensim(X9Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y9TestZmean))

% %% Test Scenario 10 - unit
% y = eigensim(X10Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y10TestUnit))
% 
% %% Test Scenario 10 - zmean
% y = eigensim(X10Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y10TestZmean))

% %% Test Scenario 11 - unit
% y = eigensim(X11Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y11TestUnit))
% 
% %% Test Scenario 11 - zmean
% y = eigensim(X11Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y11TestZmean))

% %% Test Scenario 12 - unit
% y = eigensim(X12Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y12TestUnit))
% 
% %% Test Scenario 12 - zmean
% y = eigensim(X12Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y12TestZmean))

% %% Test Scenario 13 - unit
% y = eigensim(X13Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y13TestUnit))
% 
% %% Test Scenario 13 - zmean
% y = eigensim(X13Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y13TestZmean))

% %% Test Scenario 14 - unit
% y = eigensim(X14Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y14TestUnit))
% 
% %% Test Scenario 14 - zmean
% y = eigensim(X14Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y14TestZmean))

% %% Test Scenario 15 - unit
% y = eigensim(X15Test,20,6,'unit','edc', 0.40);
% assert(isequal(y,y15TestUnit))
% 
% %% Test Scenario 15 - zmean
% y = eigensim(X15Test,20,6,'zmean','edc', 0.40);
% assert(isequal(y,y15TestZmean))