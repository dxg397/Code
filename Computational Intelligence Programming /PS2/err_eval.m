%evaluate rms error for all training patterns (rows in training_patterns)
% and targets

function [rmserr,esqd] = err_eval(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns,targets)
[noutputs,npats] =size(targets);
%esqd=0;
%evaluate all output errors
   [outputs_j,outputs_k]=eval_2layer_fdfwdnet(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,training_patterns);
   errvecs=outputs_k-targets; %column vectors of y-t
   sqd_errs = errvecs.*errvecs;
   esqd= sum(sum(sqd_errs));  %sum squared errors; have not applied 1/2 yet
  rmserr=sqrt(esqd/npats);
  %TEST:
%   [K,P] = size(targets);
%   esqd_test=0;
%   for p=1:P
%       [outputj,outputk]= eval_2layer_fdfwdnet(W1p,b1_vec,W21,b2_vec,training_patterns(:,p));
%       err = outputk-targets(:,p);
%       esqd_test=esqd_test+err'*err*0.5;
%   end
%   esqd_test
%   esqd
