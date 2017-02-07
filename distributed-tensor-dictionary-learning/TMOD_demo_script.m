clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% T-MOD demo by Florian Roemer, DVT, TU Ilmenau, Feb 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate dictionary and coefficients randomly
M = [5,7];
N = [10,11];
T = 500;
A1 = randn(M(1),N(1));
A2 = randn(M(2),N(2));
S = randn(prod(N),T); % just for demo purposes, for DL, S should be sparse...

% matrix-valued data model: A is Kronecker
X = kron(A1,A2)*S;

% tensor-valued data model: X and S can be reshaped into tensors
Xt = permute(reshape(X.', [T,M(2),M(1)]), [3,2,1]);
St = permute(reshape(S.', [T,N(2),N(1)]), [3,2,1]);
% % now, Xt = St \times_1 A2 \times_2 A2

% to see this, let us look at one- and two-mode unfoldings
Xt1 = reshape(permute(Xt,[1,3,2]),M(1),[]);
Xt2 = reshape(permute(Xt,[2,1,3]),M(2),[]);
St1 = reshape(permute(St,[1,3,2]),N(1),[]);
St2 = reshape(permute(St,[2,1,3]),N(2),[]);

% these now obey these rules:
disp(norm(Xt1 - A1*St1*kron(A2,eye(T)).'))
disp(norm(Xt2 - A2*St2*kron(eye(T),A1).'))
% % we can use this to solve for A1 and A2...
% e.g., via Xt1\(St1*kron(...)), but you might want to use a regularized
% inverse instead (as it is common in MOD...)