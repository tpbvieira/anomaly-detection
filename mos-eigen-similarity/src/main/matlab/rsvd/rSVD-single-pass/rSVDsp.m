function [U, S, V]= rSVDsp(A, k, b)
% function [U, s, V]= rSVDsp(A, k, b)
% b is the block size
% k+10 should be a multiple of b.

if nargin<3,
    b=10;
end

[m, n]= size(A);
os= 10;         % oversampling 10
k= k+os;
Omg= randn(n, k);
G= A*Omg;
H= A'*G;
Q= zeros(m, 0);
B= zeros(0, n);

s= floor(k/b);
for i=1:s,
    temp= B*Omg(:, (i-1)*b+1:i*b);
    Yi= G(:, (i-1)*b+1:i*b)- Q*temp;
    
    [Qi, Ri]= qr(Yi, 0);
%     Bi= Ri'\(H(:, (i-1)*b+1:i*b)'-Omgi'*B'*B);
    [Qi, Rit]= qr(Qi-Q*(Q'*Qi), 0);
    Ri= Rit*Ri;
    Bi= Ri'\(H(:, (i-1)*b+1:i*b)'-Yi'*Q*B-temp'*B);
    
    Q= [Q, Qi];
    B= [B; Bi];
end
B(isnan(B))=0;
B(isinf(B))=0;
[U1, S, V]= svd(B, 'econ');
U= Q*U1;
S= diag(S);
U= U(:,1:k-os); V= V(:,1:k-os);
S= S(1:k-os);
% err= norm(U*S*V'-A, 'fro')/norm(A, 'fro')
