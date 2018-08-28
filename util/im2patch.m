function [X, Idx, numPatchesY, numPatchesX] = im2patch( I, patchsize, patchspacing, offset )
%IM2PATCH Extract patches from an image according to some parameters
% patchsize: a 2-element vector of the size of each patch in [h w]
% patchspacing: either 'distinct', 'sliding' or a 2-element vector of
% spacing between patches in [y x]
% offset : a 2-element vector resembling the offset to start at
% extracting patches, zero by default
%
% @Author: Christoph Baur
    
    if nargin < 4
        offset = [0 0];
    end
    if nargin < 3
        patchspacing = [1 1];
    end
    
    % Checking if patchspacing is a string
    if ischar(patchspacing)
        if strcmp(patchspacing, 'sliding')
            patchspacing = [1 1];
        elseif strcmp(patchspacing, 'distinct')
            patchspacing = patchsize;
        else
            error('Invalid patch spacing. Make sure it is either sliding, distinct or [h w]!');
        end
    end
    
    % In case there was an offset provided, crop the new image
    if offset(1) > 0 && offset(2) > 0
        I = I(offset(1):end, offset(2):end);
    end
    
    [imageHeight imageWidth] = size(I);
    numPatchesY = ceil(imageHeight / patchspacing(1));
    numPatchesX = ceil(imageWidth / patchspacing(2));
    % Catch edge case: when spacing is 1px in a direction, we have to
    % correct for that
    if patchspacing(1) == 1
        numPatchesY = numPatchesY - patchsize(1) + 1;
    end
    if patchspacing(2) == 1
        numPatchesX = numPatchesX - patchsize(2) + 1;
    end
    maxYCoordinate = (numPatchesY-1) * patchspacing(1) + 1;
    maxXCoordinate = (numPatchesX-1) * patchspacing(2) + 1;
    % And another important thing: if spacing is not one, we have to pad
    % the image with zeros
    I_padded = zeros(maxYCoordinate + patchsize(1) - 1, maxXCoordinate + patchsize(2) - 1);
    I_padded(1:imageHeight, 1:imageWidth) = I;
    
    % Either way, we extract any patches via sliding as its fast, and afterwards we
    % filter
    X_tmp = im2col(I_padded, patchsize, 'sliding');
    X = [];
    Idx = [];
    for x=1:patchspacing(2):maxXCoordinate
       for y=1:patchspacing(1):maxYCoordinate
           X(:,end+1) = X_tmp(:,(x-1)*maxYCoordinate + y);
           Idx(:,end+1) = [y x];
       end
    end
    
    % Correct the indices for the offset in case it was set
    if offset(1) > 0 && offset(2) > 0
        Idx = Idx + repmat(offset', 1, size(Idx,1));
    end
end

