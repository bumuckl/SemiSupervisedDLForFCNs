function [ images, labels ] = getBatchFromDataAndLabelsOnDisk(imdb, batch, outputscale)
%GETBATCH Pick a batch of images from the imdb struct that only holds
%filenames of patches and labels

    if nargin < 3
        outputscale = 0.25;
    end

    filenames = imdb.images.filenames(batch);
    filenames_labels = imdb.images.filenames_labels(batch);
    images = zeros(0,0,0,0);
    labels = zeros(0,0,0,0);
    for f=1:length(filenames)
       images(:,:,:,end+1) = imread([imdb.meta.pathstr '/' imdb.meta.name '/' filenames{f}]);
       labels(:,:,:,end+1) = imresize(CNN.loadLabels([imdb.meta.pathstr '/' imdb.meta.name '/' filenames_labels{f}]), outputscale, 'nearest');
    end
    images = normalize2(images, 'single', 8);
    labels = single(labels);
    
    disp(['Batch min: ' num2str(min(images(:))) ', max: ' num2str(max(images(:)))]);
end

