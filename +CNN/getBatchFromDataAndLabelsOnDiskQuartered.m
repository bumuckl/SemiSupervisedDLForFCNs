function [ images, labels ] = getBatchFromDataAndLabelsOnDiskQuartered(imdb, batch)
%GETBATCH Pick a batch of images from the imdb struct that only holds
%filenames of patches and labels

    filenames = imdb.images.filenames(batch);
    filenames_labels = imdb.images.filenames_labels(batch);
    images = zeros(0,0,0,0);
    labels = zeros(0,0,0,0);
    for f=1:length(filenames)
       img = imread([imdb.meta.pathstr '/' imdb.meta.name '/' filenames{f}]);
       lbl = imresize(CNN.loadLabels([imdb.meta.pathstr '/' imdb.meta.name '/' filenames_labels{f}]), 0.25, 'nearest');
			 
			 % Now quarter image and labels
			 sz = size(img);
             szl = size(lbl);
			 images(:,:,:,end+1) = img(1:sz(1)/2, 1:sz(2)/2); % top left quadrant
			 labels(:,:,:,end+1) = lbl(1:szl(1)/2, 1:szl(2)/2);
			 
			 images(:,:,:,end+1) = img(1:sz(1)/2, (sz(2)/2+1):end); % top right quadrant
			 labels(:,:,:,end+1) = lbl(1:szl(1)/2, (szl(2)/2+1):end);
			 
			 images(:,:,:,end+1) = img(sz(1)/2+1:end, 1:sz(2)/2); % bottom left quadrant
			 labels(:,:,:,end+1) = lbl(szl(1)/2+1:end, 1:szl(2)/2);
			 
			 images(:,:,:,end+1) = img(sz(1)/2+1:end, (sz(2)/2+1):end); % bottom right quadrant
			 labels(:,:,:,end+1) = lbl(szl(1)/2+1:end, (szl(2)/2+1):end);
    end
    images = normalize2(images, 'single', 8);
    labels = single(labels);
    
    disp(['Batch min: ' num2str(min(images(:))) ', max: ' num2str(max(images(:)))]);
end

