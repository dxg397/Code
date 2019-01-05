function premapped_outputs = compute_premapped_outputs(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,x,y);
%UNTITLED convert (x,y) into many nodal outputs over range of x and range
%of y
%premap the x value(s)
[outputs_j,outputs_k]=eval_2layer_fdfwdnet(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,x);
%premap the y_value(s)
premapped_outputs=outputs_k;
[outputs_j,outputs_k]=eval_2layer_fdfwdnet(Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,y);
premapped_outputs = [premapped_outputs;outputs_k];


end

