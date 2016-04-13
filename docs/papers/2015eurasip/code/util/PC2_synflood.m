function [v] = PC2_synflood(tcp80,tcp443,udp53,tcp600,udp67,udp68)
v = -0.0164*tcp80 + 0.9996*tcp443 + 0.0082*udp53 + 0.0213*tcp600 + 0*udp67 + 0*udp68;
