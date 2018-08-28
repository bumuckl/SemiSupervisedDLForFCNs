function inputs = getDAGNNBatch(imdb, batch)
%GETBATCH Pick a batch of images from the imdb struct

    images = imdb.images.data(:,:,:,batch);
    if ndims(imdb.images.labels) > 2
        labels = imdb.images.labels(:,:,:,batch,:);
    else
        labels = imdb.images.labels(1,batch);
    end
    inputs = {'input', images, 'label', labels} ;
end

