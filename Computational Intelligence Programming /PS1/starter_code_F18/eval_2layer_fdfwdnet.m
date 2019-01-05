%helper fnc to evaluate outputs of a 2-layer fdfwd net;
function [outputj,outputk]=eval_2layer_fdfwdnet(W1p,W21,b1_vec,b2_vec,stim_vec)
%W1p
%b1_vec
%stim_vec

u_net_layer1 = W1p*stim_vec + b1_vec;
outputj=logsig(u_net_layer1);

%interneuron outputs
%W21
%b2_vec
%outputj
u_net_layer2 = W21*outputj + b2_vec;

%output = layer_activation_fnc(u_net_layer2)
outputk=logsig(u_net_layer2);