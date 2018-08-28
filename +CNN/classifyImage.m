function responseMap = classifyImage( net, I, stepsize )
%CLASSIFY Given a trained network and an image I, classify each pixel in
%the image

    if nargin < 3
		stepsize = [8 8]; % [y x]
	end
    patchsize = net.normalization.imageSize(1:2);

    % Mirror image boundaries
    I_padded = padarray(I, floor(patchsize ./ 2), 'symmetric');
    
    % Iterate over all pixels in x and y direction and extract patches
    % around the px acting as the centerpoint
    responseMap = zeros([size(I,1), size(I,2), 2]);
    for y=1:stepsize(1):size(I,1)
       for x=1:stepsize(2):size(I,2)
           patch = patchAroundPoint( I_padded, [y,x] + floor(patchsize ./ 2), patchsize );
           res = vl_simplenn(net, patch);
           responseMap(y,x,:) = res(end).x;
       end
       %disp([y, x]);
    end
    
end

