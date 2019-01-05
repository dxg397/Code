function [est_dW_matrices] = compute_est_dW_matrices(W_matrices,phi_codes,b_vecs,stimuli,targets)
%perturb each synapse, one at a time and numerically estimate dE/dw
[L_layers,dummy] = size(W_matrices);
[dummy,P] = size(stimuli);
est_dW_matrices=cell(L_layers);
eps = 0.000001;

[all_x_vecs] = solve_all_layers(W_matrices,phi_codes,b_vecs,stimuli);
[rmserr,E0] = err_eval(all_x_vecs,targets)

for layer=1:L_layers
    W_matrices_test = W_matrices;
    dW = 0*W_matrices{layer};
    Wl_test = W_matrices{layer};
    [Irows,Jcols] = size(Wl_test);
    for irow = 1:Irows
        for jcol=1:Jcols
            Wl_test = W_matrices{layer};
            Wl_test(irow,jcol) = Wl_test(irow,jcol) + eps;
            W_matrices_test{layer} = Wl_test;
            [all_x_vecs] = solve_all_layers(W_matrices_test,phi_codes,b_vecs,stimuli);
            [rmserr,E1] = err_eval(all_x_vecs,targets);
            dEdW_est = (E1-E0)/eps;
            dW(irow,jcol)=dEdW_est;
        end
    end
    est_dW_matrices{layer} = dW;
end
    

