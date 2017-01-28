function D = get64x256(s)
% get64x256       Get a 64x256 regular dictionary (frame) 
%                 
% ex: 
% D = get64x256();         % first 64 columns are identity matrix
% D = get64x256('dct');    % first 64 columns are 2D-dct basis functions

% Just some quick and dirty code to get a tight and almost Grassmannian
% dictionary of size 64x256, i.e. mu for D is 1/8
% Note that last basis in D is not one of the bases used in get64xK.m

W2 = [1,1; 1,-1];        
W4 = 0.5*kron(W2, W2);     
V4 = 0.5*ones(4)-eye(4);   

B = reshape( get16x144(), 16, 16, 9 );  % the 9 bases in 16x144 dictionary

D4 = kron(B(:,:,7),V4);
p2 = [1 2 3 4; 3 4 1 2; 2 1 4 3; 4 3 2 1];
k1 = kron( [0,1,2,3]', ones(1,16) );
a = [0 4 8 12];       
q1 = [p2+a(1), p2+a(2), p2+a(3), p2+a(4)];
q4 = reshape((4*q1-k1),1,64);   % order of rows in D4

D = [eye(64), kron(B(:,:,2),W4), kron(B(:,:,3),V4), D4(q4,:)];

if (nargin > 0) && ischar(s) && strcmpi(s,'dct')
    % rotate so that the identity basis will be the dct-2 basis
    D = reshape(D, 8, 8, 256);
    for k=1:256; D(:,:,k) = idct2(D(:,:,k)); end;
    D = reshape(D,64,256);
end

return;

