function D = get16x144()
% get16x144       Get a 16x144 regular dictionary (frame)
%
% D = get16x144();
% p = dictprop(D);

W2 = [1,1; 1,-1];        
W4 = 0.5*kron(W2, W2);     
V4 = 0.5*ones(4)-eye(4);   
% make the 9 bases, Bn
% the file, ...\Karl\DICT\matlab\get16x144.m (03.11.2009), includes how
% these 9 bases were found.
B1 = eye(16);
B2 = kron(W4, W4);  
B3 = kron(V4, V4);
% permute the rows of B3
B4 = B3([1, 5, 13, 9,  6, 2, 10, 14,  11, 15, 7, 3,  16, 12, 4, 8], :);
B5 = B3([1, 9, 5, 13,  6, 14, 2, 10,  11, 3, 15, 7,  16, 8, 12, 4], :);
B6 = B3([1, 13, 9, 5,  6, 10, 14, 2,  11, 7, 3, 15,  16, 4, 8, 12], :);
% change sign of 4 of the columns of B5
A = B5*diag(sign(V4(:)));  
% permute rows of A
B7 = A([1, 5, 13, 9,  2, 6, 14, 10,  3, 7, 15, 11,  4, 8, 16, 12], :);
B8 = A([1, 9, 5, 13,  2, 10, 6, 14,  3, 11, 7, 15,  4, 12, 8, 16], :);
B9 = A([1, 13, 9, 5,  2, 14, 10, 6,  3, 15, 11, 7,  4, 16, 12, 8], :);
D = [B1, B2, B3, B4, B5, B6, B7, B8, B9];

return;
