%matrix2vec: converts a matrix to a vector by stacking all cols
function [Avec]=matrix2vec(A)
[nrows,ncols]=size(A);
Avec=zeros(nrows*ncols,1); %create vector of consistent size

%first, string out the matrix into a vector
Atemp=A(:,1);
for icol=2:ncols
    Atemp=[Atemp;A(:,icol)];
end

%convert to bipolar 
for i=1:nrows*ncols
    if Atemp(i)==0
        Avec(i)= -1;
    else
        Avec(i)=1;
    end
end