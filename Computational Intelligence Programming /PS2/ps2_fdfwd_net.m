%2-layer fdfwd network to be trained w/ custom BP
%for robot-arm data, define 3 inputs, including bias (I=3), 
% J interneurons (incl bias), and 2 outputs (x,y)
%choose logsig (sigmoid) nonlinear activation fnc 

clear all

nnodes_layer1=20; %ADJUST THIS: number of interneurons
%activation fncs: 1 = logsig, 2 = linear
phi1_code = 1; %set activation type to logsig for layer 1
phi2_code = 1; %set activation type to logsig for layer 2

load arm_xy.dat; %training data is stored in this file
training_patterns = (arm_xy(:,1:2))'; %pattern inputs are columns

targets = (arm_xy(:,3:4))';
%REMAP TARGETS TO RANGE 0.2 to 0.8
%when result is achieved, need to unmap z back to original offsets and
%range
zmin= min(min(targets))
zmax = max(max(targets))
z_range = zmax-zmin;
%shrink range to 0.2-> 0.8 = 0.6
targets = targets*(0.6/z_range);
zmin= min(min(targets));;
z_offset = 0.2-zmin
[zdim1,zdim2]=size(targets);
targets = targets+z_offset*ones(zdim1,zdim2);
zmax = max(max(targets))
%DEBUG: reduce the data for testing/debug; comment out lines below when
%ready
% training_patterns = training_patterns(:,1:6)
% targets = targets(:,1:6)
%DEBUG: end of comment block
xmin = min(training_patterns(1,:))
xmax = max(training_patterns(1,:))
ymin = min(training_patterns(2,:))
ymax = max(training_patterns(2,:))


[ndim_inputs,Npats]=size(training_patterns); %get pattern dimensions from data
%input_biases = ones(1,Npats);
%training_patterns = [input_biases;training_patterns];



[nnodes_layer2,dummy] = size(targets); %number of outputs
%weights from pattern inputs to layer-1 (interneurons)
%initialize weights to random numbers, plus and minus--may want to change
%range; first row has dummy inputs, since first interneuron is unchanging bias node
W1p = (2*rand(nnodes_layer1,ndim_inputs)-1); %matrix is nnodes_layer1 x ndim_inputs
W1p_new = W1p;
b1_vec = (2*rand(nnodes_layer1,1)-1); %bias vector for layer 1
b1_vec_new = b1_vec;
%weights from interneurons (and bias) to output layer 
W21 = (2*rand(nnodes_layer2,nnodes_layer1)-1); %includes row for bias inputs
W21_new = W21;
b2_vec = (2*rand(nnodes_layer2,1)-1);
b2_vec_new = b2_vec;

%eta = 0.00001 %to test derivatives and dE, use very small eta
eta=0.02; % tune this value; may also want to vary this during iterations
iteration=0;
%BP:
iter1k=0;
%delta_L = zeros(nnodes_layer2,1)
%delta_1 = zeros(nnodes_layer1,1)

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
    [dWL_cum,dW_Lminus1_cum,delta_L_cum,delta_Lminus1_cum] = compute_dW_from_sensitivities(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets);
 
    %DEBUG: uncomment the following and prove that your compute_W_derivs
    %yields the same answer as numerical estimatesfor dE/dW
    %comment out to run faster, once debugged
%   [est_dWkj,delta_L_est]= numer_est_Wkj(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets); %and numerical estimate
  %DEBUG output: sensitivities computed 2 different ways should be the same
%    dWL_cum
%    est_dWkj
%    delta_L_cum
%    delta_L_est
%    
%    dW_Lminus1_cum %display sensitivities dE/dwji
%    [est_dWji,delta_Lminus1_est]= numer_est_Wji(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets); %and numerical estimate
%    est_dWji
%    delta_Lminus1_cum
%    delta_Lminus1_est
%    pause
   %DEBUG: comment out the above lines from DEBUG to DEBUG when ready

    %use gradient descent to update all weights;
    %for debug, can try suppressing all but one of these at a time, then
    %compare dE to dE_expected
    W1p_new=W1p-eta*dW_Lminus1_cum;
    b1_vec_new =b1_vec - eta*delta_Lminus1_cum;
    W21_new=W21-eta*dWL_cum;
    b2_vec_new = b2_vec - eta*delta_L_cum;
    
    [rms,Esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets);
    [rms,Esqd_new] = err_eval(W1p_new,b1_vec_new,phi1_code,W21_new,b2_vec_new,phi2_code,training_patterns,targets);
    dE = 0.5*(Esqd_new-Esqd);
    if (dE>0) %oops; decrease step size and back up
        eta=0.5*eta;
        W1p_new=W1p-eta*dW_Lminus1_cum;
        b1_vec_new =b1_vec - eta*delta_Lminus1_cum;
        W21_new=W21-eta*dWL_cum;
        b2_vec_new = b2_vec - eta*delta_L_cum;   
    end
    %eta = eta*1.1 %increase step size
    
    
    %optional debug: plot out incremental progress every plot_iter iterations
    plot_iter=100;
    if (iteration-iter1k>plot_iter)
        %expect dE = -eta*delta_L_cum'*delta_L_cum
        [rms,Esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets);
        [rms,Esqd_new] = err_eval(W1p_new,b1_vec_new,phi1_code,W21_new,b2_vec_new,phi2_code,training_patterns,targets);
        dE = 0.5*(Esqd_new-Esqd)
        dE_expected_W21 = sum(sum((W21_new-W21).*dWL_cum))
        dE_expected_W1p = sum(sum((W1p_new-W1p).*dW_Lminus1_cum))
        dE_expected_b2 = delta_L_cum'*(b2_vec_new-b2_vec)
        dE_expected_b1 = delta_Lminus1_cum'*(b1_vec_new-b1_vec)
        dE_expected = dE_expected_W21+dE_expected_W1p+dE_expected_b2+dE_expected_b1
 
        
        figure(1)
        ffwd_surfplot(W1p,b1_vec,phi1_code,W21(1,:),b2_vec(1),phi2_code,xmin,xmax,ymin,ymax); 
        hold on
        plot3(training_patterns(1,:),training_patterns(2,:),targets(1,:),'*')
        hold off
        figure(2)
        ffwd_surfplot(W1p,b1_vec,phi1_code,W21(2,:),b2_vec(2),phi2_code,xmin,xmax,ymin,ymax);         
        iter1k=iteration;
        iteration
        [rmserr,esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets)
        hold on
        plot3(training_patterns(1,:),training_patterns(2,:),targets(2,:),'*')
        hold off  
        %it would be appropriate here to save the current weights!
        pause(1)
        %eta = input('enter eta: ') %this for debug w/ manual eta
        %adjustment
    end


end
  %ffwd_surfplot(W1p,W21); %print out final plot, if loop terminates



