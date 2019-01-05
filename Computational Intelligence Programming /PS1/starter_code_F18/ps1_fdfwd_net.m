%2-layer fdfwd network to be trained w/ custom BP
%for xor, define 2 inputs and bias values,  J interneurons, with biases,
%and 1 output, with bias
%
%choose logsig (sigmoid) nonlinear activation fnc 
% put training patterns in rows: P=4 patterns
clear all
p1=[0,0];
t1=[0];
p2=[1,1];
t2=[0];
p3=[1,0];
t3=[1];
p4=[0,1];
t4=[1];
training_patterns=[p1;p2;p3;p4];  %store pattern inputs as row vectors in a matrix
targets=[t1;t2;t3;t4];  % should have these responses; rows correspond to input pattern rows
ndim_inputs=2; %2D patterns 
nnodes_layer2=1; %single output

%EXPERIMENT WITH NEXT 2 PARAMS:
nnodes_layer1=9; %try this many interneurons; experiment with this number

eps= 0.01; % tune this value; may also want to vary this during iterations


%weights from pattern inputs to layer 1 (interneurons)
%initialize weights to random numbers, plus and minus--may want to change
%range; first row has dummy inputs, since first interneuron is unchanging bias node
W1p = (2*rand(nnodes_layer1,ndim_inputs)-1); %matrix is nnodes_layer1 x ndim_inputs

%weights from interneurons to output layer 
W21 = (2*rand(nnodes_layer2,nnodes_layer1)-1); 
%randomize bias values for interneurons:
b1_vec = (2*rand(nnodes_layer1,1)-1);
%and bias values for output neuron(s):
b2_vec = (2*rand(nnodes_layer2,1)-1);

%evaluate networkover a grid of inputs and plot using "surf"; 
%works only in special case:  assumes inputs are 2-D and range from 0 to 1 
ffwd_surfplot(W1p,W21,b1_vec,b2_vec);
iteration=0;
%BP:
iter1k=0;
%delta_L = zeros(nnodes_layer2,1)
%delta_1 = zeros(nnodes_layer1,1)
while (1>0) % infinite loop--ctl-C to stop; edit this to run finite number of times
    iteration=iteration+1;    
    %compute all derivatives of error metric w/rt all weights; put these
    %derivatives in matrices dWL_cum and dW_Lminus1_cum

    [dWL_cum,dW_Lminus1_cum,delta_L_cum,delta_Lminus1_cum] = compute_dW_from_sensitivities(W1p,W21,b1_vec,b2_vec,training_patterns,targets);    

    %DEBUG: uncomment the following and prove that your compute_W_derivs
    %yields the same answer as numerical estimates for dE/dW
    %comment out to run faster, once debugged
    %display derivative computation
    dWL_cum
    %delta_L_cum
    %do a SLOW alternative numerical approximation of the same
    %sensitivities:
    est_dWkj= numer_est_Wkj(W1p,W21,b1_vec,b2_vec,training_patterns,targets) %and numerical estimate

    %display sensitivities dE/dwji
    dW_Lminus1_cum
    %SLOW numerical estimate of L-1 layer sensitivities
    est_dWji=numer_est_Wji(W1p,W21,b1_vec,b2_vec,training_patterns,targets) %and numerical estimate

    %use gradient descent to update all weights:
    %W1p=W1p-eps*dWji;
    W1p=W1p-eps*dW_Lminus1_cum;
    %W21=W21-eps*dWkj;
    W21=W21-eps*dWL_cum;
    %and biases:
    b1_vec = b1_vec - eps*delta_Lminus1_cum;
    b2_vec = b2_vec -eps*delta_L_cum;

    
    %optional debug: plot out incremental progress every plot_iter iterations
    plot_iter=100;
    if (iteration-iter1k>plot_iter)
        ffwd_surfplot(W1p,W21,b1_vec,b2_vec); 
        iter1k=iteration;
        iteration
        pause(1)
    end
    rmserr = err_eval(W1p,W21,b1_vec,b2_vec,training_patterns,targets)
%     xlswrite('dWL_cum',dWL_cum)
%     xlswrite('est_dWkj',est_dWkj)
%     xlswrite('dW_Lminus1_cum',dW_Lminus1_cum)
%     xlswrite('est_dWji',est_dWji)

end
  ffwd_surfplot(W1p,W21,b1_vec,b2_vec); %print out final plot, if loop terminates



