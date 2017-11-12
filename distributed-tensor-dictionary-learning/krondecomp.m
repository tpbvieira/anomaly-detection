
% ---------------------------------------------------------------------------- 
% Given a matrix A (of size m1*m2, n1*n2), find the optimal kronecker
% decomposition of the matrix.
% the following procedure finds the B,C pair that minimize
% norm( A - kron(B,C), 'fro' )**2
%
% See Van Loan and Pitsianis, "Approximation with Kronecker Products."

function [B,C] = krondecomp( A, m1, m2, n1, n2 )

tmpA = rearrange( A, m1, m2, n1, n2 );
[u,a,v] = svd( tmpA );
vecB = a(1,1) * u(:,1);
vecC = v(:,1);

B = vtm( vecB, m1, n1 );
C = vtm( vecC, m2, n2 );

%norm( A - kron(B,C), 'fro' )

%
% ---------------------------------------------------------------------------- 
% The rearrange operator rearranges a matrix in a form that is suitable
% for a rank-one approximation through kronecker products.
%

% ----------------------------------------------------------------------------

function [newA] = rearrange( A, m1, m2, n1, n2 )

% there should be m1*m2 rows and n1*n2 columns
[ h,w ] = size( A );
if ( h ~= m1*m2 )
   error( 'Incorrect blocking parameters m1 m2!' );
end;
if ( w ~= n1*n2 )
   error( 'Incorrect blocking parameters n1 n2!' );
end;

for j = 1:n1
  Aj = [];
  for i = 1:m1
    tmp = mblock( A, i, j, m2, n2 );
    tmpv = mtv( tmp )';
    Aj( i, 1:length(tmpv) ) = tmpv;
  end;

  [ th, tw ] = size( Aj );
  newA( (j-1)*th+1:j*th , 1:tw ) = Aj;
end;


% ----------------------------------------------------------------------------
% get the i,j block of A (which is of size m2xn2)

function [rval] = mblock( A, i, j, m2, n2 )
  rval = A( (i-1)*m2 + 1:i*m2, (j-1)*n2+1:j*n2 );

% ----------------------------------------------------------------------------
% converts a matrix to a vector by stacking the columns

function [vector] = mtv( A )

[ h,w ] = size( A );
for ii = 1:w
  vector( (ii-1)*h+1:ii*h ) = A( : , ii );
end
vector = vector';

% ----------------------------------------------------------------------------
% converts a vector to a matrix by unstacking the columns

function [newA] = vtm( vector, h, w )

l = length( vector );
for i = 1:l/h
  newA( 1:h, i ) = vector( (i-1)*h+1:i*h );
end

% ----------------------------------------------------------------------------
