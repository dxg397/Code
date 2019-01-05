function [dE_dkvec1,dE_dbias_params] = numerical_test_dE_dk_layer1(kmaps,kvec,bmaps,b_params,W_matrices,phi_codes,b_vecs,stimuli,targets)
%this fnc is specialized for layer 1
%vary kernel components 1 at a time and estimate derivatives numerically
%start w/ given W's and b's:
%compute W and b from kernels:
[L_layers,dummy] = size(W_matrices);
[dummy,P] = size(stimuli);
[kmap_rows,kmap_cols] = size(kmaps{1});
W1 = zeros(kmap_rows,kmap_cols);
[nmaps,dummy] = size(kmaps); %how many maps do we have?  should be same as
dE_dkvec1=zeros(nmaps,1);
eps = 0.000001;

%build a W matrix as the weighted sum of kernel maps
%this W matrix operations on a strung-out image vector, and it yields a
%strung-out feature map
for i=1:nmaps
    W1 = W1+kvec(i)*kmaps{i};
end

bvec1 = zeros(kmap_rows,1);
b_params_layer1 = b_params{1};
[num_b_params,dummy]= size(b_params_layer1) %expect 1-D
for i=1:num_b_params
    bvec1 = bvec1+b_params_layer1(i)*bmaps{i};
end
%%bvec1
%b_vecs{1}
dE_dbias_params=zeros(num_b_params,1); 

%compute the system error for this W matrix:
W_matrices_test = W_matrices;
W_matrices_test{1} = W1;
[all_x_vecs] = solve_all_layers(W_matrices_test,phi_codes,b_vecs,stimuli);
[rmserr,E0] = err_eval(all_x_vecs,targets);

%vary bias params to get dE/db terms:
for bb=1:num_b_params
    bparam = b_params_layer1(bb);
    bparam = bparam+eps;
    b_params_test = b_params_layer1;
    b_params_test(bb) = bparam;
    bvec1 = zeros(kmap_rows,1);
    for i=1:num_b_params
        bvec1 = bvec1+b_params_test(i)*bmaps{i};
    end    
    b_vecs_test = b_vecs;
    b_vecs_test{1} = bvec1;
    [all_x_vecs] = solve_all_layers(W_matrices,phi_codes,b_vecs_test,stimuli);
    [rmserr,E1] = err_eval(all_x_vecs,targets);
    dE_db =(E1-E0)/eps; 
    dE_dbias_params(bb) = dE_db
end

%now vary kernel params one at a time:
for kk=1:nmaps
    W_matrices_test = W_matrices;
    kernel_test = kvec;
    kernel_test(kk)=kernel_test(kk)+eps;
    W1 = zeros(kmap_rows,kmap_cols);
    for i=1:nmaps
        W1 = W1+kernel_test(i)*kmaps{i};
    end    
    
    W_matrices_test{1} = W1;
    [all_x_vecs] = solve_all_layers(W_matrices_test,phi_codes,b_vecs,stimuli);
    [rmserr,E1] = err_eval(all_x_vecs,targets);
    dE_dk =(E1-E0)/eps;
    dE_dkvec1(kk) = dE_dk; 
end
