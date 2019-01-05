function [W,b_vec] = hard_coded_thresholding_weights(num_neurons)
%here is a fnc to hard-code weights for the purpose of applying a threshold
%to each input
wscale=10;
bscale=5;
W = eye(num_neurons)*wscale;
b_vec = -bscale*ones(num_neurons,1);
end

