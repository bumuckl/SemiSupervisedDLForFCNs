function responseMap = classifyImage3( net, I )
%CLASSIFY Given a trained network and an image I, classify each pixel in
%the image. This version just adds a special pooling layer at the beginning
%of the network in order to classify each pixel
    
    if nargin < 3
        stepsize = 1;
    end
    patchsize = net.normalization.imageSize(1:2);

    % Mirror image boundaries
    I_padded = padarray(I, floor(patchsize ./ 2), 'symmetric');
    
    % TODO
    
    responseMap = zeros(size(I,1), size(I,2));
end

