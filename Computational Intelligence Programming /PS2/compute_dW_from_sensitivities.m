%computes derivatives w/rt weights for BP
function  [dWL_cum,dW_Lminus1_cum,delta_L_cum,delta_Lminus1_cum] = compute_dW_from_sensitivities(Wji,bj_vec,phi_code1,Wkj,bk_vec,phi_code2,training_patterns,targets)

[K,P] =size(targets); %dim of output vec and num training patterns; size()
                      % K=2 P=200
[J,I] = size(Wji); %J=interneurons
delta_L_cum = zeros(K,1); 
delta_L = delta_L_cum; %set same size   delta_L= (dE/db)T
delta_Lminus1_cum = zeros(J,1);
delta_Lminus1 = delta_Lminus1_cum; %%init size

dW_Lminus1=Wji*0;
dWL_cum = Wkj*0; 
dWL = Wkj*0;
dW_Lminus1_cum=Wji*0;

%dW_Lminus1 = zeros(J,I);



    %need to compute outputs of both j and k layers for all stimulus
    %patterns
    [outputs_j,outputs_k]=eval_2layer_fdfwdnet(Wji,bj_vec,phi_code1,Wkj,bk_vec,phi_code2,training_patterns);
    err_vecs = outputs_k - targets;
    phi_prime_L_vecs = fnc_phi_prime(phi_code2,outputs_k); %make sure this is consistent w/ act. fnc.
    
    
    deltas_L = phi_prime_L_vecs.*err_vecs; %FIX ME!!! compute delta_L for every pattern excitation
        
    
    delta_L_cum= sum(deltas_L,2); 
           
    %compute remaining delta_l(n) recursively:
    phi_prime_Lminus1_vecs = fnc_phi_prime(phi_code1,outputs_j);%outputs_j.*(1-outputs_j); 
   
        for p=1:P
           delta_Lminus1(:,p)= Wkj'*deltas_L(:,p).*phi_prime_Lminus1_vecs(:,p);
        end                       %FIX ME!!! put in recursive relationship
   % end  

    delta_Lminus1_cum= sum(delta_Lminus1,2); %net bias sensitivities for this layer; add all columns
    
    %given all deltas(n) for all layers, can compute synapse sensitivities
     dW_Lminus1=Wkj'*deltas_L(:,1).*phi_prime_Lminus1_vecs(:,1)*training_patterns(:,1)';
     tempdW_Lminus1=dW_Lminus1;
     dWL=deltas_L(:,1)*outputs_j(:,1)';
     tempdWL=dWL;
    
    for p=2:P
   
        %layer L synapse sensitivities:
         %FIX ME!!
          dWL=deltas_L(:,p)*outputs_j(:,p)';
          dWL_cum = dWL_cum + dWL;
        %layer L-1 synapse sensitivities
        %FIX ME!!      
        dW_Lminus1=Wkj'*deltas_L(:,p).*phi_prime_Lminus1_vecs(:,p)*training_patterns(:,p)';
       dW_Lminus1_cum = dW_Lminus1_cum+dW_Lminus1; 
       %could make this a loop for arbitrary number of layers...
    end
    dWL_cum=dWL_cum+tempdWL;
    dW_Lminus1_cum=dW_Lminus1_cum+tempdW_Lminus1;
    
    
    
    
      
      

