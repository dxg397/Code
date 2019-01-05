%evaluate rms error for all training patterns (rows in training_patterns)
% and targets

function [rmserr,esqd] = err_eval(all_x_vectors,targets)
[noutputs,npats] =size(targets);
[L_layers,dummy] = size(all_x_vectors);
%evaluate all output errors
   outputs = all_x_vectors{L_layers};
   errvecs=outputs-targets; %column vectors of y-t
   sqd_errs = errvecs.*errvecs;
   esqd= 0.5*sum(sum(sqd_errs));  %sum squared errors; have not applied 1/2 yet
  rmserr=sqrt(esqd/npats);
