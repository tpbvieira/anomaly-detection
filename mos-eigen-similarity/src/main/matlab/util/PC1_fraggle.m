function [v] = PC1_fraggle(tcp80,tcp443,tcp53,udp19,udp67,udp68)
v = 0.0044*tcp80 + 0.0002*tcp443 + 0.0001*tcp53 - 1.0000*udp19 + 0*udp67 + 0*udp68;
