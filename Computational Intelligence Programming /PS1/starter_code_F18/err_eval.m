%evaluate rms error for all training patterns (rows in training_patterns)
% and targets

function [rmserr,esqd] = err_eval(W1p,W21,b1_vec,b2_vec,training_patterns,targets)
temp =size(targets);
npats=temp(1);
noutputs=temp(2);

esqd=0;
%evaluate all output errors
for i=1:npats
   stim_vec=training_patterns(i,:)';
   [outputj,outputk]=eval_2layer_fdfwdnet(W1p,W21,b1_vec,b2_vec,stim_vec);
   errvec=outputk-targets(i,:);
   esqd=esqd+errvec'*errvec;
end
       
rmserr=sqrt(esqd/npats);
