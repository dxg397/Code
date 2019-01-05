%this main function tests [ kmaps ] = fnc_kmaps(kernel,image), which
%converts convolution into a matrix multiply
%This is tested by comparing the result to Matlab's conv2()
%Note, however, the difference between "correlation" with a kernel vs.
%"convolution" with a kernel.  Definition of convolution uses a flipped
%version of the correlation kernel

%invent random images and kernels: choose the dimensions
nrows_image = 3;
ncols_image = 4;
L_layers=2 %define this as a 2-layer network
%Krows_kernel = 2;
%Kcols_kernel = 3;
%generate or load training data;
%each image has a Krows x Kcols "feature" embedded.
%the location of this feature (upper left corner) in the image is encoded
%in the corresponding target (1 in this location, 0 elsewhere)
[stimuli,targets,Krows_kernel,Kcols_kernel] = gen_training_data(nrows_image,ncols_image);
[image_vec_dim,npatterns] = size(stimuli)
%extract one of the training vectors and reshape it as an image
image = reshape(stimuli(:,1)',ncols_image,nrows_image)'
output='paused...'
pause

%for synapses to layer 1, make this a convolutional layer with the
%following kernel:
kernel = rand(Krows_kernel,Kcols_kernel)
%here is the main deal: express convolution in terms of a set of injection
%maps (this is for layer1 only)
[ kmaps ] = fnc_kmaps(kernel,image);
[nmaps,dummy]=size(kmaps)
for kk=1:nmaps
    output='kmap'
    kmaps{kk}
    output='paused'
    pause
end
bmaps=cell(L_layers); %need to fill this in
bparams = cell(L_layers); %need to fill this in

kernel_vec = SOH(kernel) %express kernel as a row vector
[kmap_rows,kmap_cols] = size(kmaps{1});
W1 = zeros(kmap_rows,kmap_cols);
[nmaps,dummy] = size(kmaps) %how many maps do we have?  should be same as
%number of kernel elements
%build a W matrix as the weighted sum of kernel maps

%FIX ME!!!
%this W matrix operations on a strung-out image vector, and it yields a
%strung-out feature map

W1 %here is the synapse matrix that is equivalent to convolution
%also want to apply a bias.  For convolution, each output gets same bias
%rationale for choice below: for image inputs in range [0,1], would like
%to convert to bipolar, so subtract 0.5 from each pixel.  But at layer1,
%inputs are bigger than this, scaled up by number of kernel components.  So
%apply a negative bias that is scaled up similarly
%b1_vec = -0.5*nmaps*ones(kmap_rows,1)
b1_map = ones(kmap_rows,1); %same bias vector to all nodes of layer 1
bmaps{1} = b1_map;
b1_param = -0.5*nmaps; %have only a single parameter for the bias kernel
bparams{1} = b1_param;
b1_vec = b1_map*b1_param; %sum of b1 maps * b1 params...trivial for 1-D
conv_vec = W1*(SOH(image))' %operate on the strung-out image
x1_vec = conv_vec; %%spiking rates of feature map, as a vector = layer1 outputs
%turn the result vector back into a corresponding 2D matrix feature map:
feature_map = reshape(conv_vec,[ncols_image-Kcols_kernel+1,nrows_image-Krows_kernel+1])'

%test: compare this result to Matlab's 2D convolution:
%BUT convolution is defined with a "flipped" kernel, so flip it before
%using conv2D:
orig_kernel = kernel
%Flip it left to right using fliplr(), then flip it top to bottom with flipud().
conv_kernel = fliplr(kernel);
conv_kernel = flipud(conv_kernel) %
%compare the result below to the technique using equivalent W matrix:
%'valid' option means don't let the kernel slide off the image; retain only
%legal values for kernel placements
conv2D_map = conv2(image,conv_kernel,'valid')
%compare these:
output='result of conv2d, SOH: '
SOH(conv2D_map)
output='result of W1*(SOH(image1)): '
conv_vec'
output='paused...'
pause

%choose weights hard-coded for pattern-match evaluation
[num_layer2_neurons,num_inputs] = size(W1)
[W2,b2_vec] = hard_coded_thresholding_weights(num_layer2_neurons)

%have a network.  Evaluate it for all input patterns:
phi1_code=1; %logsig
phi2_code=1; %logsig
x1_vecs = eval_1layer_fdfwdnet(W1,b1_vec,phi1_code,stimuli)
x2_vecs = eval_1layer_fdfwdnet(W2,b2_vec,phi2_code,x1_vecs)

%define the network in arbitrary-length cells:
%fill W_matrices,phi_codes,b_vecs
W_matrices = cell(2,1);
W_matrices{1} = W1;
W_matrices{2} = W2;
phi_codes = cell(2,1);
phi_codes{1}=1;
phi_codes{2}=1; %logsig
b_vecs = cell(2,1);
b_vecs{1} = b1_vec;
b_vecs{2} = b2_vec;

%how do the x2_vecs compare to the target vecs?
err_vecs = targets-x2_vecs

%compute all the sensitivity vectors:
[bias_sensitivity_vecs,all_x_vecs] = compute_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets)
%test this: do bias perturbations and check resulting E
[num_est_bias_sensitivities] = compute_num_est_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets);

% [rmserr,E0] = err_eval(all_x_vecs,targets)
% [num_est_bias_sensitivities] = compute_num_est_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets)
% %try a perturbation of db in b2_vec:
% eps=0.000001;
% test_vec = b2_vec;
% test_vec(1) = test_vec(1)+eps;
% b_vecs{2}=test_vec;
% [bias_sensitivity_vecs,all_x_vecs] = compute_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets)
% [rmserr,E1] = err_eval(all_x_vecs,targets)
% numerical_dE_db = (E1-E0)/eps
% delta_vecs_L = bias_sensitivity_vecs{2};
% delta_vec_cum = sum(delta_vecs_L,2)
%%FINISH TESTING: test all terms of delta_vec_cum, both layers
[L_layers,dummy] = size(num_est_bias_sensitivities);
for layer=1:L_layers
    this_layer = layer
    num_bias_sensitivities = num_est_bias_sensitivities{layer};
    num_est_delta_cum = sum(num_bias_sensitivities,2);
    analytic_delta = sum(bias_sensitivity_vecs{layer},2);
    [nrows,dummy] = size(num_est_delta_cum);
    for row = 1:nrows
        dEdbi = analytic_delta(row);
        est_dEdbi= num_est_delta_cum(row);
        compare_dEdbi = [dEdbi,est_dEdbi]
    end    
end
%bias sensitivities check out
%compute sensitivities with respect to synapses:
[dW_matrices] = compute_dW_from_sensitivities(bias_sensitivity_vecs,all_x_vecs,stimuli);
[est_dW_matrices] = compute_est_dW_matrices(W_matrices,phi_codes,b_vecs,stimuli,targets);
%compare to numerical estimates of dE/dw
for layer=1:L_layers
    this_layer = layer
    dW = dW_matrices{layer};
    dW_est = est_dW_matrices{layer};
    [dW_rows,dW_cols] = size(dW);
    for irow=1:dW_rows
        for jcol = 1:dW_cols
           num_est_dw = dW_est(irow,jcol);
           analytic_dw= dW(irow,jcol);
           compare = [analytic_dw,num_est_dw]
           %pause
        end
    end
    diff = dW_est-dW
    %ans = input('done w/ layer; enter 1 to continue')
end
%dbias and dW sensitivities look good; now convert to kernel sensitivities
%focus on layer 1; kmaps corresponds to layer1 only;
%use corresponding sensitivities of synapses of layer 1
dE_dW1 = dW_matrices{1};
[dE_dkvec] = compute_dE_dkvec(kmaps,dE_dW1)
%test this numerically:
[dE_dkvec1_est,dE_dbias_params] = numerical_test_dE_dk_layer1(kmaps,kernel_vec,bmaps,bparams,W_matrices,phi_codes,b_vecs,stimuli,targets)

%looks good;  also need sensitivity of biases for layer1
dE_db_param_layer1 = sum(sum(bias_sensitivity_vecs{1},2))
%sensitivity w/rt scalar bias param for layer1 looks good

%iterate to learn kernel to hit targets:
eta = 0.01 %learning rate
[bias_sensitivity_vecs,all_x_vecs] = compute_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets);
%compute system error:
[rmserr,E_old] = err_eval(all_x_vecs,targets)
iter=0;
iter_pause=100; %pause after this many iterations
while(1)
    %simulate network and compute sensitivites for layer1:
    [bias_sensitivity_vecs,all_x_vecs] = compute_bias_sensitivity_vecs(W_matrices,phi_codes,b_vecs,stimuli,targets);
    %compute system error:
    [rmserr,E_new] = err_eval(all_x_vecs,targets)
    if (E_new<E_old)
        %error decreased...accept updates
        eta = 1.1*eta   
        if eta>1
            eta=1;
        end
        E_old = E_new
        kernel_vec = kernel_vec_new; %accept the new params
        b1_param = b1_param_new; %        
    else
      eta = 0.5*eta
      E_old
    end
    %compute synapse sensitivities
    [dW_matrices] = compute_dW_from_sensitivities(bias_sensitivity_vecs,all_x_vecs,stimuli);
    dE_dW1 = dW_matrices{1};
    %compute kernel-param sensitivites for layer1
    [dE_dkvec] = compute_dE_dkvec(kmaps,dE_dW1)'
    %compute bias sensitivity for layer1
    dE_db_param_layer1 = sum(sum(bias_sensitivity_vecs{1},2))
    %kernel:
    kernel_vec_new = kernel_vec -eta*dE_dkvec
    %bias:
    b1_param_new= b1_param - eta*dE_db_param_layer1
    %build W1 and b1vec from kernels and maps
    W1 = 0*dE_dW1; %sets size of W1
    for i=1:nmaps
        W1 = W1+kernel_vec_new(i)*kmaps{i};
    end
    %build b1vec:
    b1_vec = b1_map*b1_param_new;
    %install these in cells:
    W_matrices{1} = W1;
    b_vecs{1} = b1_vec;
   % pause
    %consider if should pause for display
    iter=iter+1;
    if iter>iter_pause
        iter=0;
        eta_val = eta
        E = E_old
        pause(1)
    end
        
end
