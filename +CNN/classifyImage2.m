function responseMap = classifyImage2( net, I, stepsize )
%CLASSIFY Given a trained network and an image I, classify each pixel in
%the image
    
    if nargin < 3
        stepsize = 1;
    end
    patchsize = net.normalization.imageSize(1:2);

    % Mirror image boundaries
    I_padded = padarray(I, floor(patchsize ./ 2), 'symmetric');
    
    % Algorithm: Divide image into blocks and process each block separately
%     responseMap = blockproc(I_padded, blocksize, @hlp_classifyBlock, 'BorderSize', floor(patchsize ./ 2));
%     
%     function result = hlp_classifyBlock(blockstruct)
%         result = CNN.classifyImage( net, blockstruct.data, [1 1] );
%     end
    
    responseMap = zeros(size(I,1), size(I,2));
    for x=1:stepsize:size(I,2)
       %endx = min(size(I,2),x+stepsize+patchsize(2)-2);
       %if endx - x + 1 < stepsize+patchsize(2)-2
       %    x = x - (stepsize + patchsize(2) - 2 - endx + x -1);
       %end
       vectorized_column = im2colstep(double(I_padded(:,x:min(size(I_padded,2),x+stepsize+patchsize(2)-2),:)), [patchsize(1),patchsize(2),3]);
       data = zeros(patchsize(1),patchsize(2),3,size(vectorized_column,2));
       for i=1:size(vectorized_column,2)
          data(:,:,:,i) = reshape(vectorized_column(:,i),patchsize(1),patchsize(2),3);
          %figure(1), cla, 
          %subplot(1,2,1), imshow(squeeze(data(:,:,:,i)));
          %subplot(1,2,2), imshow(I_padded);
          %disp(num2str(i));
       end
       data = single(data);
       res = vl_simplenn(net, data);
       %responseMap(:,x) = res(end).x(:,:,2,:);
       res = squeeze(res(end).x(:,:,2,:));
       responseMap(:,x:x+floor(length(res)/size(I,1))-1) = reshape(res, size(I,1), floor(length(res)/size(I,1)) );
       disp(num2str(x));
    end
end

