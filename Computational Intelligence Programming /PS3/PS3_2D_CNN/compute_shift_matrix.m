function [ shift_op_trans ] = compute_shift_matrix( SOH_image_dim )
%create a matrix that performs a shift right
shift_op = eye(SOH_image_dim,SOH_image_dim);
shift_op = shift_op(1:SOH_image_dim-1,:);
shift_op = [zeros(1,SOH_image_dim);shift_op];
shift_op_trans = shift_op'; %use this to compute: row_shifted + row*shift_op_trans
end

