%2-layer fdfwd network to be trained w/ custom BP
%for robot-arm data, define 3 inputs, including bias (I=3), 
% J interneurons (incl bias), and 2 outputs (x,y)
%choose logsig (sigmoid) nonlinear activation fnc 

clear all
load arm_xy.dat; %training data is stored in this file
training_patterns = (arm_xy(:,1:2))'; %pattern inputs are columns

%ADJUST THIS: number of interneurons AFTER premapping
nnodes_layer1=50; 
phi1_code=1; %logsig
phi2_code=2; %linear  %NOTICE THIS: LINEAR ACTIVATION FNC FOR LAST LAYER (OK)

targets = (arm_xy(:,3:4))';
%DEBUG: reduce the data for testing/debug; comment out lines below when
%ready
% training_patterns = training_patterns(:,1:6)
% targets = targets(:,1:6)
%DEBUG: end of comment block
Npreset_nodes=11;
xmin = min(training_patterns(1,:))
xmax = max(training_patterns(1,:))
ymin = min(training_patterns(2,:))
ymax = max(training_patterns(2,:))
[Wx1,bvec_x1,Wx2,bvec_x2]=compute_preset_weights(xmin,xmax,Npreset_nodes);
[Wy1,bvec_y1,Wy2,bvec_y2]=compute_preset_weights(ymin,ymax,Npreset_nodes);
phix1_code=1; %logsig
phix2_code=1; %logsig
phiy1_code=1; %logsig
phiy2_code=1; %logsig

%preset_targets_x = eye(Nx1,Nx1);
%[outputs_j,outputs_k]=eval_preset_fdfwdnet(W1,bvec_x1,W2,bvec_x2,preset_inputs_x)

dx=(xmax-xmin)/(Npreset_nodes-1);
test_inputs_x=xmin:dx*0.2:xmax;
%[outputs_j,outputs_k]=eval_preset_fdfwdnet(Wx1,bvec_x1,Wx2,bvec_x2,test_inputs_x)
[outputs_j,outputs_k]=eval_2layer_fdfwdnet(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,test_inputs_x);
figure(1)
plot(test_inputs_x,outputs_k)
title('premapped nodal outputs')
xlabel('x')
ylabel('node output')

dy=(ymax-ymin)/(Npreset_nodes-1);
test_inputs_y=ymin:dy*0.2:ymax;
[outputs_j,outputs_k]=eval_2layer_fdfwdnet(Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,test_inputs_y);
figure(2)
plot(test_inputs_y,outputs_k)

%this is how to compute premappings of inputs (x,y) onto a vector of nodes
premapped_outputs = compute_premapped_outputs(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,test_inputs_x,test_inputs_y);
[N_premapped_nodes,N_test_samples] = size(premapped_outputs)

%perform premapping over all training inputs:
%this expands (x,y) inputs into (e.g.) 22 nodal outputs
[ndim_inputs,Npats]=size(training_patterns); %get pattern dimensions from data
premapped_training_patterns = [];
for npat = 1:Npats
    x = training_patterns(1,npat);
    y = training_patterns(2,npat);
    premapped_outputs = compute_premapped_outputs(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,x,y);
    premapped_training_patterns=[premapped_training_patterns,premapped_outputs];
end
[ndim_inputs,N_patterns] = size(premapped_training_patterns);

%now, append 2 layers to compute output z1; z2 will get its own network
z1_targets = targets(1,:);



[nnodes_layer2,dummy] = size(z1_targets) %number of outputs; this will just be "1"
%weights from premapped pattern inputs to layer 1 (interneurons)
%initialize weights to random numbers, plus and minus--may want to change
%range; first row has dummy inputs, since first interneuron is unchanging bias node
%reduce range of w's for larger number of inputs
W1p = 1/sqrt(nnodes_layer1)*(2*rand(nnodes_layer1,ndim_inputs)-1); %matrix is nnodes_layer1 x ndim_inputs
W1p_new = W1p;
b1_vec = 1/sqrt(nnodes_layer1)*(2*rand(nnodes_layer1,1)-1); %bias vector for layer 1
b1_vec_new = b1_vec;
%weights from interneurons (and bias) to output layer 
W21 = 1/sqrt(nnodes_layer2)*(2*rand(nnodes_layer2,nnodes_layer1)-1); %includes row for bias inputs
W21_new = W21;
b2_vec = 1/sqrt(nnodes_layer2)*(2*rand(nnodes_layer2,1)-1);
b2_vec_new = b2_vec;
%evaluate networkover a grid of inputs and plot using "surf"; 
%works only in special case:  assumes inputs are 2-D and range from 0 to 1 
%ffwd_surfplot(W1p,W21);
eta=0.001; % tune this value; may also want to vary this during iterations
iteration=0;
%BP:
iter1k=0;

while (1>0) % infinite loop--ctl-C to stop; edit this to run finite number of times
    %compute all derivatives of error metric w/rt all weights; put these
    %derivatives in matrices dWkj and dWji
    %use the "delta" vectors for each row to compute recursively
    iteration=iteration+1;
    b2_vec = b2_vec_new; %update bias weights, layer 2
    b1_vec = b1_vec_new;
    W21 = W21_new;
    W1p = W1p_new;
    %FIX THIS FUNCTION...
    [dWL_cum,dW_Lminus1_cum,delta_L_cum,delta_Lminus1_cum] = compute_dW_from_sensitivities(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_training_patterns,z1_targets);
 
    %DEBUG: uncomment the following and prove that your compute_W_derivs
    %yields the same answer as numerical estimatesfor dE/dW
    %comment out to run faster, once debugged
%     [est_dWkj,delta_L_est]= numer_est_Wkj(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_training_patterns,z1_targets); %and numerical estimate
%   %DEBUG output: sensitivities computed 2 different ways should be the same
%     dWL_cum
%     est_dWkj
%     delta_L_cum
%     delta_L_est
% % %    s
%     dW_Lminus1_cum %display sensitivities dE/dwji
%     [est_dWji,delta_Lminus1_est]= numer_est_Wji(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_training_patterns,z1_targets); %and numerical estimate
%     est_dWji
%     delta_Lminus1_cum
%     delta_Lminus1_est
  
   %DEBUG: comment out the above lines from DEBUG to DEBUG when ready

    %use gradient descent to update all weights:
    W1p_new=W1p-eta*dW_Lminus1_cum;
    b1_vec_new =b1_vec - eta*delta_Lminus1_cum;
    W21_new=W21-eta*dWL_cum;
    b2_vec_new = b2_vec - eta*delta_L_cum;
    [rms,Esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_training_patterns,z1_targets);
    [rms,Esqd_new] = err_eval(W1p_new,b1_vec_new,phi1_code,W21_new,b2_vec_new,phi2_code,premapped_training_patterns,z1_targets);
    dE = 0.5*(Esqd_new-Esqd);
    if (dE>0) %oops; decrease step size and back up
        eta=0.5*eta;
        W1p_new=W1p-eta*dW_Lminus1_cum;
        b1_vec_new =b1_vec - eta*delta_Lminus1_cum;
        W21_new=W21-eta*dWL_cum;
        b2_vec_new = b2_vec - eta*delta_L_cum;   
    end
    eta = eta*1.1 %increase step size
    
    
    %optional debug: plot out incremental progress every plot_iter iterations
    plot_iter=100;
    if (iteration-iter1k>plot_iter)
        %expect dE = -eta*delta_L_cum'*delta_L_cum
        [rms,Esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_training_patterns,z1_targets);
        [rms,Esqd_new] = err_eval(W1p_new,b1_vec_new,phi1_code,W21_new,b2_vec_new,phi2_code,premapped_training_patterns,z1_targets);
        dE = 0.5*(Esqd_new-Esqd)
        dE_expected_W21 = sum(sum((W21_new-W21).*dWL_cum))
        dE_expected_W1p = sum(sum((W1p_new-W1p).*dW_Lminus1_cum))
        dE_expected_b2 = delta_L_cum'*(b2_vec_new-b2_vec)
        dE_expected_b1 = delta_Lminus1_cum'*(b1_vec_new-b1_vec)
        dE_expected = dE_expected_W21+dE_expected_W1p+dE_expected_b2+dE_expected_b1
 
        
%         figure(1)
%premapped_outputs = compute_premapped_outputs(Wx1,bvec_x1,Wx2,bvec_x2,Wy1,bvec_y1,Wy2,bvec_y2,test_inputs_x,test_inputs_y);
          ffwd_surfplot_w_premap(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,xmin,xmax,ymin,ymax); 
         hold on
         plot3(training_patterns(1,:),training_patterns(2,:),targets(1,:),'*')
         hold off
   
        iter1k=iteration;
        iteration
        [rmserr,esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_training_patterns,z1_targets)

        pause(1)
       % %eta = input('enter eta: ')
    end


end
  %ffwd_surfplot(W1p,W21); %print out final plot, if loop terminates



