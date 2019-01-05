function [num_est_bias_sensitivities] = compute_num_est_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets)
%vary one parameter of bias vectors at a time to estimate numerical
%derivatives
[L_layers,dummy] = size(W_matrices) %W_matrices has each W matrix stored cell by cell
num_est_bias_sensitivities = cell(L_layers,1);

eps = 0.000001;
[dim_outputs,P] =size(targets); %dim of output vec and num training patterns
for layer=L_layers:-1:1
    Wl = W_matrices{layer};
    bl_vec = b_vecs{layer};
    phi_code = phi_codes{layer};
    [n_neurons_layer_l,n_neurons_layer_lminus1] = size(Wl);
    num_est_deltas_l = zeros(n_neurons_layer_l,1);
    [all_x_vecs] = solve_all_layers(W_matrices,phi_codes,b_vecs,stimuli);
    xl_vecs = all_x_vecs{layer};
    [rmserr,E0] = err_eval(all_x_vecs,targets)
    b_vecs_test = b_vecs;
    for i_neuron=1:n_neurons_layer_l
        bvec_test = bl_vec;
        bvec_test(i_neuron) = bvec_test(i_neuron) + eps;
        b_vecs_test{layer} = bvec_test;
        [all_x_vecs_test] = solve_all_layers(W_matrices,phi_codes,b_vecs_test,stimuli);
        [rmserr,E1] = err_eval(all_x_vecs_test,targets);
        numerical_dE_db = (E1-E0)/eps;
        num_est_deltas_l(i_neuron) = numerical_dE_db;
    end
      num_est_bias_sensitivities{layer}=num_est_deltas_l;
end
    
        

