%function to evaluate a 2-input, single-output, 2-layer feedforward network
%and plot it as a surface plot
% surf(x,y,Z) and surf(x,y,Z,C), with two vector arguments replacing
%     the first two matrix arguments, must have length(x) = n and
%     length(y) = m where [m,n] = size(Z).  In this case, the vertices
%     of the surface patches are the triples (x(j), y(i), Z(i,j)).
%     Note that x corresponds to the columns of Z and y corresponds to
%     the rows.
function ffwd_surfplot_w_premap(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,xmin,xmax,ymin,ymax)
dx = (xmax-xmin)/10;
dy = (ymax-ymin)/10;
xvals=[xmin:dx:xmax];
yvals=[ymin:dy:ymax];
Z=zeros(11,11); %holder for 11x11 grid of outputs
for (i=1:11)
    for(j=1:11)
        x = xvals(i);
        y = yvals(j);
        premapped_outputs = compute_premapped_outputs(Wx1,bvec_x1,phix1_code,Wx2,bvec_x2,phix2_code,Wy1,bvec_y1,phiy1_code,Wy2,bvec_y2,phiy2_code,x,y);
        %stim = [xvals(i);yvals(j)]; %stimulate network at this set of inputs, including bias
        [outputj,outputk] = eval_2layer_fdfwdnet(W1p,b1_vec,phi1_code,W21,b2_vec,phi2_code,premapped_outputs);
     Z(j,i)= outputk(1);
    end
end
surf(xvals,yvals,Z)