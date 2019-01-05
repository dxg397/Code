%helper fnc to evaluate outputs of a 2-layer fdfwd net;
%this version treats the bias inputs as a separate vector (instead of
%including a dummy neuron of output "1") 
function [outputs_j,outputs_k]=eval_2layer_fdfwdnet(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,stimuli)
%size(W1p)
%stimuli
%size(b1_vec)
[ninputs,npats] = size(stimuli);
u_net_vecs_layer1 = W1p*stimuli+b1_vec*ones(1,npats);
outputs_j= fnc_phi(phi1_code,u_net_vecs_layer1); %logsig(u_net_vecs_layer1);

%augment interneuron outputs with addl bias term:
u_net_vecs_layer2 = W21*outputs_j+b2_vec*ones(1,npats);

%output = layer_activation_fnc(u_net_layer2)
outputs_k=fnc_phi(phi2_code,u_net_vecs_layer2); %linear_activation_fnc(u_net_vecs_layer2); %try linear activation fnc