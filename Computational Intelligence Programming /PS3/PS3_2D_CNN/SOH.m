function [ soh_vec ] = SOH(A )
%string-out horizontally; convert a matrix to a row vector, appending rows
%sequentially
[nrows,ncols]=size(A);
A_trans = A';

soh_vec = reshape(A_trans,nrows*ncols,1)';

end

