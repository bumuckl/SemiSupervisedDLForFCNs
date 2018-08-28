% Intermediate function for getting batches of images and their labels
function inputs = fn_getBatchDAGNNUnsupervised(imdb, batch)
	images = imdb.images.data(:,:,:,batch);
    if ndims(imdb.images.labels) > 2
        labels = imdb.images.labels(:,:,:,batch,:);
    else
        labels = imdb.images.labels(1,batch);
    end
    
    mode = unique(imdb.images.set(batch));
    labels_all = labels;
    if mode == 1 % Only in training mode add the unlabeled data
        % Sample "unlabeled" patches and fill the second data blob
        numUnlabeledItems = round(imdb.images.lu_ratio * numel(batch));
        batch_unlabeled = randperm(size(imdb.images.data_unlabeled, 4), numUnlabeledItems);
        images = cat(4, images, imdb.images.data_unlabeled(:,:,:,batch_unlabeled));
        if ndims(imdb.images.labels_unlabeled) > 2
            labels_all = cat(4, labels, imdb.images.labels_unlabeled(:,:,:,batch_unlabeled,:));
            labels = cat(4, labels, NaN(size(imdb.images.labels_unlabeled(:,:,:,batch_unlabeled,:))));
            %domains = cat(4, domains, 2*ones(size(imdb.images.labels_unlabeled(:,:,:,batch_unlabeled,:))));
        else
            labels_all = [labels; imdb.images.labels_unlabeled(1,batch_unlabeled)];
            labels = [labels; NaN(size(imdb.images.labels_unlabeled(1,batch_unlabeled)))];
            %domains = [domains; 2*ones(size(imdb.images.labels_unlabeled(1,batch_unlabeled)))];
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