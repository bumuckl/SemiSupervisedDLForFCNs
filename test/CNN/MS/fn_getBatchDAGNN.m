% Intermediate function for getting batches of images and their labels
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