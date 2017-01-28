function D = get64xK(s)
% get64xK         Get a 64xK regular dictionary (frame), now K=9*64 
%                 
% ex: 
% D = get64xK();         % first 64 columns are identity matrix
% D = get64xK('dct');    % first 64 columns are 2D-dct basis functions
% p = dictprop(D);

P = perms(1:8);
B = reshape(get16x144(), 16, 16, 9);
K = 9*64;

W2 = [1,1; 1,-1];        
W4 = 0.5*kron(W2, W2);     
V4 = 0.5*ones(4)-eye(4);   

D = zeros(64,64,K/64);
D(:,:,1) = eye(64);
D(:,:,2) = kron( kron(W4, W4), W4);   % B(:,:,2) = kron(W4, W4); 
D(:,:,3) = kron( kron(V4, V4), V4);   % B(:,:,3) = kron(V4, V4); 
% the next 6 can all be found by permuting rows in a matrix which is 
% the Kronecker product of a basis in B=get16x144() and V4
% the numbers can be found by ~/Matlab/dle/m10.m  (UiS unix system, user: karlsk)
% many other possiblities exist!
D(:,:,4) = pRows8and8( kron(B(:,:,7), V4), P( 6294,:), P(35325,:) ); 
D(:,:,5) = pRows8and8( kron(B(:,:,3), V4), P(19038,:), P(36805,:) ); 
D(:,:,6) = pRows8and8( kron(B(:,:,6), V4), P(22873,:), P(12104,:) ); 
D(:,:,7) = pRows8and8( kron(B(:,:,4), V4), P(24589,:), P(15640,:) ); 
D(:,:,8) = pRows8and8( kron(B(:,:,4), V4), P(17593,:), P( 6491,:) ); 
D(:,:,9) = pRows8and8( kron(B(:,:,5), V4), P(28468,:), P( 4195,:) ); 

D = reshape(D, 64, K);

if (nargin > 0) && ischar(s) && strcmpi(s,'dct')
    % rotate so that the identity basis will be the dct-2 basis
    D = reshape(D, 8, 8, K);
    for k=1:K; D(:,:,k) = idct2(D(:,:,k)); end;
    D = reshape(D,64,K);
end

return;

% permute the rows of A as given by p1 and p2
function C = pRows8and8(A,p1,p2)
% p1 and p2 are permutations of 1:8, a row of perms(1:8)
p2 = (p2-1)*8;   % permutations of: 0,8,16,24,32,40,48,56
q2 = [p1+p2(1), p1+p2(2), p1+p2(3), p1+p2(4), ...
      p1+p2(5), p1+p2(6), p1+p2(7), p1+p2(8)];
C = A(q2,:);
return


