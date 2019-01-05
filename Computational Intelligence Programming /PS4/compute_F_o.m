%function compute_F_o--compute ordered derivative F_o at time t
%return vector dF/dsigma at time t
function [F_o_vec] = compute_F_o(W,errs,F_u,t)
temp = size(F_u);
Nneurons = temp(1);
T_time_steps=temp(2);
F_o_vec = zeros(Nneurons,1);

%step through all nodes, except bias and input;
%update all ordered derivs w/rt sigmas at time step t
% for t< final time, dE/dsigma(t) depends on dE/du(t+1)
if t<T_time_steps   
    F_u_vec=F_u(:,t+1);
	for i = 3:Nneurons
        for j = 3:Nneurons
            F_o_vec(i)=F_o_vec(i)+F_u_vec(j)*W(j,i);
        end
    end
   % F_o_vec = F_u_vec; %THIS IS A LIE; FIX IT
end

F_o_vec(1)=0; %don't change bias = neuron 1
F_o_vec(2)=0; %don't change input = neuron 2
%for output node, neuron 3, add in influences of partial deriv of errors w/rt sigmas at time t
for i = 3:Nneurons
    %add in influences of partial deriv of errors w/rt sigmas at time t
    F_o_vec(i) = F_o_vec(i) + 2*errs(t);
end
