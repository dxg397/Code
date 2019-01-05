function [all_x_vecs] = solve_all_layers(W_matrices,phi_codes,b_vecs,stimuli)
x_vecs_l_minus_1 = stimuli; %inputs to first layer
[L_layers,dummy] = size(W_matrices);
all_x_vecs = cell(L_layers,1);
for layer=1:L_layers
    W = W_matrices{layer};
    b_vec = b_vecs{layer};
    phi_code = phi_codes{layer};
    %debug:
    layer_num = layer
    sizeW = size(W)
    size_bvec = size(b_vec)
    size_xvec = size(x_vecs_l_minus_1)
    x_vecs_l = eval_1layer_fdfwdnet(W,b_vec,phi_code,x_vecs_l_minus_1);
    all_x_vecs{layer}=x_vecs_l;
    x_vecs_l_minus_1 = x_vecs_l; %get ready for next layer stimulation
end