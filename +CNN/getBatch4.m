function [ images, labels ] = getBatch4(imdb, batch)
%GETBATCH Pick a batch of images from the imdb struct that only hols
%filenames of patches

% @Shadi Albarqouni, TU Munich, Germany

    filenames = imdb.images.filenames(batch);
    images = zeros(0,0,0,0);
    for f=1:length(filenames)
       images(:,:,:,end+1) = imread([imdb.meta.pathstr '/' imdb.meta.name '/' filenames{f}]); 
    end
    images = normalize2(images, 'single', 8);
    %labels = imdb.images.labels(1,batch);
    labels = imdb.images.labels(:,:,:,batch,:); 
    
    disp(['Batch min: ' num2str(min(images(:))) ', max: ' num2str(max(images(:)))]);
end

