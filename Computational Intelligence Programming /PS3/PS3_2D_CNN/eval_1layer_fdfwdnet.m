%helper fnc to evaluate outputs of a 2-layer fdfwd net;
%this version treats the bias inputs as a separate vector (instead of
%including a dummy neuron of output "1") 
function [outputs]=eval_1layer_fdfwdnet(W,b_vec,phi_code,stimuli)
%size(W1p)
%stimuli
%size(b1_vec)
[ninputs,npats] = size(stimuli);
u_net_vecs = W*stimuli+b_vec*ones(1,npats);
outputs= fnc_phi(phi_code,u_net_vecs); %logsig(u_net_vecs_layer1);
