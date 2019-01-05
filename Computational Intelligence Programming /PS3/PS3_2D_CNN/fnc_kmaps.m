function [ kmaps ] = fnc_kmaps(kernel,image)
%for 2-D kernel and 2-D image, compute corresponding maps such that
%W = sum(kernel(i))*kmaps(i);
%image will be SOH; compute length:
[nrows,ncols] = size(image);
SOH_image_dim = nrows*ncols;
[Krows,Kcols] = size(kernel);
SOH_kernel_dim = Krows*Kcols;
kmaps = cell(SOH_kernel_dim,1); %this many maps from kernel to W
[ shift_op_trans ] = compute_shift_matrix( SOH_image_dim );

%first kernel component map, first row:
zero_row = zeros(1,SOH_image_dim);
start_row = zero_row;
start_row(1)=1;
Map_row=start_row;
%make a map from this first row:
map_num=1;
[ kmap ] = fnc_kmap(kernel,image,Map_row);
kmaps{1} = kmap;
%shift this Kcols-1 times to represent all params of first row of kernel
for col_shift=2:Kcols
    Map_row = Map_row*shift_op_trans;
    map_num=map_num+1;    
    [ kmap ] = fnc_kmap(kernel,image,Map_row);
    kmaps{map_num} = kmap;
end
%repeat this for remaining kernel rows
%note that, for each new kernel row, need to shift selection
%of image pixel to end of an image row, then shift addl by kernel column
%e.g., for 2nd kernel row:
Map_row = zero_row;
Map_row(1)=1;
%shift to account for next line of image:
for kernel_row=2:Krows
  Map_row=start_row;
  for col_shift=1:ncols*(kernel_row-1)
    Map_row = Map_row*shift_op_trans;
  end
  %now process all kernel components on row kernel_row:
  %shift this Kcols-1 times to represent all params of first row of kernel
    for col_shift=1:Kcols
        map_num=map_num+1;        
        [ kmap ] = fnc_kmap(kernel,image,Map_row);
        kmaps{map_num} = kmap;
        Map_row = Map_row*shift_op_trans;
    end  
end

