function [v] = PC1_synflood(tcp80,tcp443,udp53,tcp600,udp67,udp68)
v = 0.0242*tcp80 - 0.0290*tcp443 + 0.0006*udp53 + 0.9995*tcp600 + 0*udp67 + 0*udp68;
