function [W1,bvec_1,W2,bvec_2]=compute_preset_weights(xmin,xmax,Npreset_nodes)
%make a network that maps input value into Npreset_nodes ranges
dx=(xmax-xmin)/(Npreset_nodes-1);

%set weights on inputs manually
w_scale = 2/dx;
W1 = w_scale*ones(Npreset_nodes,1);
bvec_1 = w_scale*(-xmin*ones(Npreset_nodes,1)-dx*(0:Npreset_nodes-1)');

w_scale2 = 2;
W2 = eye(Npreset_nodes,Npreset_nodes);
for i=1:Npreset_nodes-1
    W2(i,i+1)=-2;
end
W2 = 20*W2;
bvec_2 = -4*ones(Npreset_nodes,1);
end

