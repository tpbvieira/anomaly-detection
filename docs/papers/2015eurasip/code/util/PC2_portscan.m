function [v] = PC2_portscan(tcp80,tcp443,udp53,tcp21,tcp22,tcp23,tcp25,tcp110,tcp143,tcp161,udp69,udp123,udp445)
v = 0.2770*tcp80 + 0.6243*tcp443 + 0.7251*udp53 + 0.0281*tcp21 + 0.0281*tcp22 + 0.0281*tcp23 + 0.0281*tcp25 + 0.0281*tcp110 + 0.0281*tcp143 + 0.0281*tcp161+ 0.0281*udp69 + 0.0281*udp123 + 0.0281*udp445;

