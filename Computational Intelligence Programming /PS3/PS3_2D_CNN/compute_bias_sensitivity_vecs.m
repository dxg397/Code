%compute bias sensitivity vectors for each layer and for each stimulus
%pattern
%
function  [bias_sensitivity_vecs,all_x_vecs] = compute_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets)
[L_layers,dummy] = size(W_matrices); %W_matrices has each W matrix stored cell by cell
bias_sensitivity_vecs = cell(L_layers,1);

[dim_outputs,P] =size(targets); %dim of output vec and num training patterns

%all_x_vecs = cell(L_layers);
%solve the network for every pattern and store all x_vecs for all layers:
[all_x_vecs] = solve_all_layers(W_matrices,phi_codes,b_vecs,stimuli);

x_vecs_L = all_x_vecs{L_layers}; %these are the output vectors for all stimuli
%start defining delta vecs from definition for layer L:
phi_prime_L_vecs = fnc_phi_prime(phi_codes{L_layers},x_vecs_L);
err_vecs = x_vecs_L - targets;
deltas_L = phi_prime_L_vecs.*err_vecs;
bias_sensitivity_vecs{L_layers} = deltas_L;
deltas_lp1 = deltas_L;
%now apply recursion:
for l = L_layers-1:-1:1
  %FIX ME!!!!
    x = all_x_vecs{l};
    phi_prime_l_vecs = fnc_phi_prime(phi_codes{l}, x);
    deltas_l = (W_matrices{l+1})'*deltas_lp1.*phi_prime_l_vecs;
    deltas_lp1 = deltas_l;
    bias_sensitivity_vecs{l} = deltas_l;
end
