function [ phi_of_u ] = fnc_phi(phi_code,inputs)
%compute activation fnc, dependent on coded type
if (phi_code==1) %logsig
    phi_of_u = logsig(inputs); 
    return;
end
if (phi_code==2) %linear
  phi_of_u = inputs;
  return;
end
output('activation function code not recognized!!')

end

