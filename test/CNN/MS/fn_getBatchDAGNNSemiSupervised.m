% Intermediate function for getting batches of images and their labels
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
function inputs = fn_getBatchDAGNNSemiSupervised(imdb, batch)
	images = imdb.images.data(:,:,:,batch);
    if ndims(imdb.images.labels) > 2
        labels = imdb.images.labels(:,:,:,batch,:);
    else
        labels = imdb.images.labels(1,batch);
    end
    
    mode = unique(imdb.images.set(batch));
    labels_all = labels;
    if mode == 1 % Only in trainign mode add the unlabeled data
        % Sample "unlabeled" patches and fill the second data blob
        numUnlabeledItems = round(imdb.images.lu_ratio * numel(batch));
        batch_unlabeled = randperm(size(imdb.images.data_unlabeled, 4), numUnlabeledItems);
        images = cat(4, images, imdb.images.data_unlabeled(:,:,:,batch_unlabeled));
        if ndims(imdb.images.labels_unlabeled) > 2
            labels_all = cat(4, labels, imdb.images.labels_unlabeled(:,:,:,batch_unlabeled,:));
            labels = cat(4, labels, NaN(size(imdb.images.labels_unlabeled(:,:,:,batch_unlabeled,:))));
        else
            labels_all = [labels; imdb.images.labels_unlabeled(1,batch_unlabeled)];
            labels = [labels; NaN(size(imdb.images.labels_unlabeled(1,batch_unlabeled)))];
        end
    end
    
    inputs = {};
    inputs{end+1} = 'input';
	inputs{end+1} = images;
	inputs{end+1} = 'label';
	inputs{end+1} = labels;
    inputs{end+1} = 'label_lu';
	inputs{end+1} = labels_all;
end