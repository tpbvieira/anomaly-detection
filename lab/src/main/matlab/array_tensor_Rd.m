function A = array_tensor_Rd(mu,M)

% ARRAY_TENSOR_RD   Array steering tensor for R-D hypercuboidal array
%
% A = ARRAY_TENSOR_RD(MU,M) returns the array steering tensor
% for an R-D hypercuboidal array. Here R = LENGTH(M) and the 
% array has M(r) sensors in the r-th mode, r=1..R.
% MU contains the spatial frequencies and is of size R x d,
% where d is the number of sources. Each element of MU should
% be in [-pi, pi].
% The resulting array steering tensor A will be of
% size M(1) x M(2) x ... x M(R) x d.

R = size(mu,1);
d = size(mu,2);
if length(M) ~= R
    error('M should be a vector of size R, mu should be R x d.');
end

for i = 1:d
    Ac = cell(1,R);
    for r = 1:R
        Ac{r} = exp(j*mu(r,i)*((0:M(r)-1)'));
    end
    A(:,i) = Nkron(Ac(R:-1:1));
end
        
A = reshape(A,[M,d]); 
% Normalizing the steering vectors
