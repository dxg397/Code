%do numerical estimation of derivatives dEsqd/dWkj
function  [dWkj, delta_L_est]= numer_est_Wkj(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets)
size(targets)
[K,P] =size(targets); %get size of output vec and number of training patterns
[J,I] = size(W1p); %get dim of inputs and dim of interneurons
[rmserr,esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets);
dWkj=0*W21;
delta_L_est=0*b2_vec;
eps=0.000001;
%Wkj
for k=1:K
    for j=1:J
       % j
        Wtemp=W21;
        Wtemp(k,j)=Wtemp(k,j)+eps;
       % Wtemp
        [rmserr,esqd2]=err_eval(W1p,b1_vec,phi1_code,Wtemp,b2_vec,phi2_code,training_patterns,targets);
        dout=esqd2-esqd;
        dWkj(k,j)=dout/eps;
    end
end
dWkj=0.5*dWkj; %deriv defined w/rt 1/2 * dEsqd/dwkj

[rmserr,esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets);
for k=1:K
    bL_vec_temp = b2_vec;
    bL_vec_temp(k) = bL_vec_temp(k)+eps;
    [rmserr,esqd2]=err_eval(W1p,b1_vec,phi1_code,W21,bL_vec_temp,phi2_code,training_patterns,targets);
     dout=esqd2-esqd;
     delta_L_est(k)=dout/eps;
end
delta_L_est = delta_L_est*0.5; %deriv defined w/rt 1/2 * dEsqd/duk
%delta_L_est
