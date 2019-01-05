%do numerical estimation of derivatives dEsqd/dWji
function  [dWji,delta_Lminus1_est]= numer_est_Wji(Wji,b1_vec,phi1_code,Wkj,b2_vec,phi2_code,training_patterns,targets)
temp =size(targets);
[K,P] =size(targets); %get size of output vec and number of training patterns
[J,I] = size(Wji); %get dim of inputs and dim of interneurons
[rmserr,esqd] = err_eval(Wji,b1_vec,phi1_code,Wkj,b2_vec,phi2_code,training_patterns,targets);

dWji=0*Wji; %set size of the W sensitivity matrix
eps=0.000001;
%Wji
for j=1:J
    for i=1:I
       % j
        Wtemp=Wji;
        Wtemp(j,i)=Wtemp(j,i)+eps;
       % Wtemp
        [rmserr2,esqd2]=err_eval(Wtemp,b1_vec,phi1_code,Wkj,b2_vec,phi2_code,training_patterns,targets);
        dout=esqd2-esqd;
        dWji(j,i)=dout/eps;
    end
end
dWji=0.5*dWji; %deriv defined w/rt 1/2 * dEsqd/dwji

delta_Lminus1_est = zeros(J,1);
[rmserr,esqd] = err_eval(Wji,b1_vec,phi1_code,Wkj,b2_vec,phi2_code,training_patterns,targets);
for j=1:J
    b1_vec_temp = b1_vec;
    b1_vec_temp(j) = b1_vec_temp(j)+eps;
    [rmserr,esqd2] = err_eval(Wji,b1_vec_temp,phi1_code,Wkj,b2_vec,phi2_code,training_patterns,targets);
     dout=esqd2-esqd;
     delta_Lminus1_est(j)=dout/eps;
end
delta_Lminus1_est = delta_Lminus1_est*0.5; %deriv defined w/rt 1/2 * dEsqd/duk
