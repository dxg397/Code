%function to evaluate a 2-input, single-output, 2-layer feedforward network
%and plot it as a surface plot
function ffwd_surfplot(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,xmin,xmax,ymin,ymax)
dx = (xmax-xmin)/10;
dy = (ymax-ymin)/10;
xvals=[xmin:dx:xmax];
yvals=[ymin:dy:ymax];
Z=zeros(11,11); %holder for 11x11 grid of outputs
for (i=1:11)
    for(j=1:11)
        stim = [xvals(i);yvals(j)]; %stimulate network at this set of inputs, including bias
        [outputj,outputk] = eval_2layer_fdfwdnet(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,stim);
     Z(j,i)= outputk(1);
    end
end
surf(xvals,yvals,Z)