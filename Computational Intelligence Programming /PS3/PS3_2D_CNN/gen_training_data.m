function [stimuli,target_vecs,Krows,Kcols] = gen_training_data(nrows_image,ncols_image)
%generate Nimages images that contain patterns, and corresponding feature maps as
%targets
nrows_image
ncols_image
Nimages = 20; %will want a lot more input images than this
%try a feature like this:
k_feature = [1 1 1 1 0 0;
             0,1,0,1,1 1;
             0,1,0,0,1 1;
             1,0,0,0,0 1;
             0,1,0,1,1 0] %a Tetris-style block
[Krows,Kcols] = size(k_feature)
stimuli = [];
target_vecs = [];
max_feature_row = nrows_image-(Krows-1)
max_feature_col = ncols_image-(Kcols-1)
for n=1:Nimages
    image = rand(nrows_image,ncols_image);
    feature_row=round((max_feature_row-1)*rand())+1
    feature_col=round((max_feature_col-1)*rand())+1
    %install the feature:
    for kr=0:Krows-1
        for kc=0:Kcols-1
            image(feature_row+kr,feature_col+kc)= k_feature(kr+1,kc+1);
        end
    end
    image
    image_vec = SOH(image)';
    stimuli=[stimuli,SOH(image)'];
    target = zeros(max_feature_row,max_feature_col);
    target(feature_row,feature_col)=1 
    target_vecs = [target_vecs,SOH(target)'];
end

