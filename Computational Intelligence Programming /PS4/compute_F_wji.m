%ordered derivatives with respect to wji
function [F_wji] = compute_F_wji(sigma_history,F_u)
temp = size(F_u);
Nneurons = temp(1);
T_time_steps=temp(2);
F_wji = zeros(Nneurons,Nneurons); %create holder for output with correct dimensions

%compute sum of F_uvals terms over time:
for j=1:Nneurons
    for i=1:Nneurons
        for t=2:(T_time_steps)
            F_wji(j,i)=F_wji(j,i) + F_u(j,t) * sigma_history(i,t-1);
        end
    end
end