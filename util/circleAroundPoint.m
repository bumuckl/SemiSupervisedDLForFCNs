function mask = circleAroundPoint( I, centerpoint, radius, numWhitePixels )
%CIRCLEAROUNDPOINT Draw a circle with the specified radius around a 
% centerpoint = [cy cx] onto a binary mask. If numWhitePixels (a simple
% scalar) is specified, the circle is filtered afterwards to contain only
% numWhitePixels pixels - these pixels are chosen randomly
    
    [W,H] = meshgrid(1:size(I,2),1:size(I,1));
    mask = sqrt((W-centerpoint(2)).^2 + (H-centerpoint(1)).^2) < radius;
    
    if nargin == 4
        idx_of_white_pixels = find(mask == 1);
        if numWhitePixels > length(idx_of_white_pixels)
            numWhitePixels = length(idx_of_white_pixels);
        end
        numBlackPixels = length(idx_of_white_pixels) - numWhitePixels;
        idx_of_white_pixels_shuffled = randperm(length(idx_of_white_pixels));
        idx_of_black_pixels = idx_of_white_pixels(idx_of_white_pixels_shuffled(1:numBlackPixels));
        mask(idx_of_black_pixels) = 0;
    end
    
    mask = I | mask;
end