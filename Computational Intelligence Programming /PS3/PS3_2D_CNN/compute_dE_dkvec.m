function [dE_dkvec] = compute_dE_dkvec(kmaps,dE_dW)
%given a matrix of synapse sensitivities, dE_dW, and mapping matrices for
%connectivity of kernel parameters to synapses (for a given layer), compute
%the net effect dE/dk by accumulating all of the relevant dE_dW for each
%kernel parameter
%here's a desirable improvement: should compute mapmat ONCE and re-use it; it never changes
mapmat =[];
[n_kmaps,dummy] = size(kmaps);
for kk=1:n_kmaps
    %make a mapping matrix consisting of rows of SOH(kmap}
    mapmat = [mapmat;SOH(kmaps{kk})];
end

%here is the key line: dE_dkvec depends on mapmat and dE_dW

dE_dkvec = mapmat* SOH(dE_dW)';


