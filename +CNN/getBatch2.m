function [ images, labels ] = getBatch2(imdb, batch)
%GETBATCH Pick a batch of images from the imdb struct

    actual_imdb = IMDB.load(imdb.images.subimdbs{batch(1)});
    images = actual_imdb.images.data;
    labels = actual_imdb.images.labels;
    
    disp(['Batch min: ' num2str(min(images(:))) ', max: ' num2str(max(images(:)))]);
end

