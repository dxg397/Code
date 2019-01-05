%function to evaluate a 2-input, single-output, 2-layer feedforward network
%and plot it as a surface plot
function ffwd_surfplot(W1p,W21,b1_vec,b2_vec)
xvals=[0:0.1:1];
yvals=[0:0.1:1];
Z=zeros(11,11); %holder for 11x11 grid of outputs
for (i=1:11)
    for(j=1:11)
        stim = [xvals(i);yvals(j)]; %stimulate network at this set of inputs, including bias
        [outputj,outputk] = eval_2layer_fdfwdnet(W1p,W21,b1_vec,b2_vec,stim);
     Z(i,j)= outputk(1);
    end
end
surf(xvals,yvals,Z)