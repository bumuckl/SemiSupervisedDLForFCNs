% Intermediate function for getting batches of images and their labels
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
function inputs = fn_getBatchDAGNNTwoInputs(imdb, batch)
	inputs = CNN.getDAGNNBatch(imdb, batch);
    
    % Sample "unlabeled" patches and fill the second data blob
    numUnlabeledItems = round(imdb.images.lu_ratio * numel(batch));
    batch_unlabeled = randperm(size(imdb.images.data_unlabeled, 4), numUnlabeledItems);
    images_unlabeled = imdb.images.data_unlabeled(:,:,:,batch_unlabeled);
    if ndims(imdb.images.labels_unlabeled) > 2
        labels_unlabeled = imdb.images.labels_unlabeled(:,:,:,batch_unlabeled,:);
    else
        labels_unlabeled = imdb.images.labels_unlabeled(1,batch_unlabeled);
    end
    
    inputs{end+1} = 'input2';
	inputs{end+1} = images_unlabeled;
	inputs{end+1} = 'labels2';
	inputs{end+1} = labels_unlabeled;
end