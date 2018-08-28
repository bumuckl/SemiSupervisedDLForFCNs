% Intermediate function for getting batches of images and their labels
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