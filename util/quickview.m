function h = quickview( tensor )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    h = figure;
    for i=1:size(tensor,4)
      img = squeeze(tensor(:,:,1,i));
      subplot(size(tensor,4),1,i), imagesc(img);
    end
    drawnow;
end

