%computes derivatives w/rt weights for BP
function  [dW_matrices] = compute_dW_from_sensitivities(bias_sensitivity_vecs,all_x_vecs,training_patterns)
[L_layers,dummy] = size(bias_sensitivity_vecs); %W_matrices has each W matrix stored cell by cell
dW_matrices = cell(L_layers,1);
[dim_outputs,P] =size(training_patterns); %dim of output vec and num training patterns

    %given all deltas(n) for all layers, can compute synapse sensitivities
    %dW_L(n) = delta_L(n) * x_lminus1(n)'
    %dW_Lminus1 = sum_n delta_Lminus1(n)*input(n)
    %start w/ first layer, for which x_lminus1 is the training inputs
    deltas = bias_sensitivity_vecs{1};
    dW = deltas(:,1)*training_patterns(:,1)';
    dW_cum = dW;
    %FIX ME!!! add a loop to accumulate effects of all stimulus patterns
    for stim=2:P
        dW = deltas(:,stim) * training_patterns(:,stim)';
        dW_cum = dW_cum + dW;
    end
    
    dW_matrices{1} = dW_cum;
    
    %all the rest of the layers:
    for layer=2:L_layers
        %FIX ME!!! install correct computation
        deltas = bias_sensitivity_vecs{layer};
        dW = deltas(:,1)*all_x_vecs{layer - 1}(:,1)';
        dW_cum = dW;
        [P, dummy] = size(all_x_vecs);
        for p=2:P
            dW = deltas(:,p)*all_x_vecs{layer - 1}(:,p)';
            dW_cum = dW_cum + dW;
        end
        dW_matrices{layer} = dW_cum;
    end
    
