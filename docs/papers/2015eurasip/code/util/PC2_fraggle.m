function [v] = PC2_fraggle(tcp80,tcp443,tcp53,udp19,udp67,udp68)
v = 0.9999*tcp80 + 0.0086*tcp443 + 0.0124*tcp53 + 0.0044*udp19 - 0*udp67 - 0*udp68;
