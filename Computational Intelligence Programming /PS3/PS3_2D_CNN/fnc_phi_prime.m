function [ phi_prime ] = fnc_phi_prime(phi_code,outputs)
%compute slope of activation fnc, dependent on coded type
if (phi_code==1) %logsig
    phi_prime = outputs.*(1-outputs);
    return;
end
if (phi_code==2) %linear
  [N_outputs,dummy] = size(outputs);
  phi_prime = ones(N_outputs,1);
  return;
end
output('activation function code not recognized!!')


end

