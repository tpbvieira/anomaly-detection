function [v] = PC1_portscan(tcp80,tcp443,udp53,tcp21,tcp22,tcp23,tcp25,tcp110,tcp143,tcp161,udp69,udp123,udp445)
v = 0.0345*tcp80 + 0.0488*tcp443 + 0.0667*udp53 - 0.3150*tcp21 - 0.3150*tcp22 - 0.3150*tcp23 - 0.3150*tcp25 - 0.3150*tcp110 - 0.3150*tcp143 - 0.3150*tcp161- 0.3150*udp69 - 0.3150*udp123 - 0.3150*udp445;
