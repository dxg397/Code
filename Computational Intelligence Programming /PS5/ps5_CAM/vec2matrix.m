%vec2mat: converts a vector to a matrix by extracting cols of length nrows
%and inserting them as parallel columns in a matrix
%need to know the dimensions of the desired matrix, nrows x ncols
function [Amat]=vec2matrix(Avec,nrows,ncols)
Amat=zeros(nrows,ncols);
%convert bipolar vector to logic (0,1)
for i=1:nrows*ncols
    if Avec(i)<0
        Avec(i)=0;
    end
end
%now, chop up vector and stack sections side by side to make matrix
for icol=0:ncols-1
    tempvec=Avec(icol*nrows+1:(icol+1)*nrows);
    Amat(:,icol+1)=tempvec;
end