function [ images, labels ] = getBatch5(imdb, batch)
%GETBATCH Pick a batch of images from the imdb struct

    images = imdb.images.data;
    labels = imdb.images.labels; 
    
end