function [ kmap ] = fnc_kmap(kernel,image,first_row)
%for 2-D kernel and 2-D image, compute corresponding map such that
%W = sum(kernel(i))*kmaps(i);
%for this helper func, provide the first row of kmap, corresponding to
%a kernel component of interest.  The rest of the map will be computed
%image will be SOH, transposed to be a column vector; compute length:
[nrows,ncols] = size(image);
SOH_image_dim = nrows*ncols;
[Krows,Kcols] = size(kernel);
SOH_kernel_dim = Krows*Kcols;
n_map_rows = (nrows-Krows+1)*(ncols-Kcols+1);
kmap = zeros(n_map_rows,SOH_image_dim); 
%first kernel component map, first row:
Map_row = first_row;
%shift this ncols-Kcols times, and repeat nrows-Krows times

%circshift is not working on my old version of Matlab, so make
% a matrix to do this.
% shift_op = eye(SOH_image_dim,SOH_image_dim);
% shift_op = shift_op(1:SOH_image_dim-1,:);
% shift_op = [zeros(1,SOH_image_dim);shift_op];
% shift_op_trans = shift_op';
[ shift_op_trans ] = compute_shift_matrix( SOH_image_dim );

kmap(1,:)= first_row;
irow=0;
for row_shift = 1:nrows-Krows+1 
  for col_shift = 1:ncols-Kcols+1
        irow = irow+1;
        kmap(irow,:)=   Map_row;
        Map_row =Map_row*shift_op_trans;
  end
  %shift right by Kcols-1
  for (k=1:Kcols-1)
        Map_row =Map_row*shift_op_trans;
  end
end
%  kmap
