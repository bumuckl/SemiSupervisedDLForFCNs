% Intermediate function for getting batches of images and their labels
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
function inputs = fn_getBatchDAGNN(imdb, batch, outputscale)
    if nargin < 3
        outputscale = 1;
    end
    
    if isfield(imdb.images, 'filenames')
        filenames = imdb.images.filenames(batch);

        images = zeros(0,0,0,0);
        labels = zeros(0,0,0,0);
        for f=1:length(filenames)
           load([imdb.meta.pathstr '/' imdb.meta.name '/' filenames{f}]);
           images(:,:,:,end+1) = patch;
           labels(:,:,:,end+1) = imresize(labels, outputscale, 'nearest');
        end
        images = single(images);
        labels = single(labels);
    else
        images = imdb.images.data(:,:,:,batch);
        labels = imdb.images.labels(:,:,:,batch);
    end
    
    inputs = {'input', images, 'label', labels} ;
end