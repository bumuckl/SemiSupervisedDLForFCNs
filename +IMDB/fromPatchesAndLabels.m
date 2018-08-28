function imdb = fromPatchesAndLabels(patches, labels, set_partitions)
%FROMPATCHESANDLABELS Convert patches and labels returned from
%"CreatePatchesData" into an IMDB struct (i.e. used by MatConvNet)
%
% @Author: Christoph Baur
    
    % Gain access to all methods of the package
    import IMDB.*

    if nargin < 3
        set_partitions = [0.6 0.2 0.2];
    end
    labels_ = [];
    for f=1:length(patches)
        for p=1:length(patches{f})
            labels_(end+1) = labels{f}{p};
        end
    end

    imdb = init();
    imdb.meta.classes = unique(labels_);
    imdb.images.data = zeros(0,0,0,0); % If this step is omitted, matlab does some weird indexing things 
    
    % We create [60 20 20] partitions while preserving the label distribution
    label_distributions = [];
    for l=1:length(imdb.meta.classes)
        label_distributions(1, l) = sum(imdb.meta.classes(l) == labels_);
    end
    label_counts = zeros(1, length(label_distributions));
    label_indexoffset = 0;
    if min(imdb.meta.classes) ~= 1
        label_indexoffset = -min(imdb.meta.classes) + 1;
        imdb.meta.classes = imdb.meta.classes + label_indexoffset; % Correct the classes
    end
    
    % Iterate over all images and labels
    for f=1:length(patches)
        for p=1:length(patches{f})
            imdb.images.data(:,:,:,end+1) = single(patches{f}{p});
            imdb.images.labels(1,end+1) = single(labels{f}{p} + label_indexoffset);
            
            if label_counts(labels{f}{p} + label_indexoffset) < set_partitions(1) * double(label_distributions(labels{f}{p} + label_indexoffset))
                imdb.images.set(1, end+1) = 1;
            elseif label_counts(labels{f}{p} + label_indexoffset) < (set_partitions(1) + set_partitions(2)) * double(label_distributions(labels{f}{p} + label_indexoffset))
                imdb.images.set(1, end+1) = 2;
            else
                imdb.images.set(1, end+1) = 3;
            end
            label_counts(labels{f}{p} + label_indexoffset) = label_counts(labels{f}{p} + label_indexoffset) + 1;
        end
    end
    
    % Normalize image data
    imdb.images.data = normalize2(imdb.images.data, 'single', 8); % Normalization to squeeze values between 0 and 1
        
    % Mean training data
    imdb.images.dataMean = mean(imdb.images.data(:,:,:,imdb.images.set == 1), 4);
end

