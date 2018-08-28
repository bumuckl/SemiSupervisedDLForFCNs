function [imdb, pickedIdx] = pickRandomSamples( imdb, numSamples )
%PICKRANDOMSAMPLES Given an existing imdb struct, pick random samples
%according to numSamples. If numSamples is a scalar, simply numSamples will
%be picked, regardless of their set. The ratio of their labels will be
%kept. If numSamples is a vector, then each entry describes how many
%samples should be picked from each class. If there are not enough labels,
%oversampling will occur and a warning is displayed.
% Note: labels have to be greater or equal than 1
%
% @Author: Christoph Baur

    % Gain access to all methods of the package
    import IMDB.*
        
    % Catch errors
    if length(numSamples) <= 0
       error('IMDB.pickRandomSamples: numSamples was not specified or empty');
    end
    if length(numSamples) > length(imdb.meta.classes)
       error('IMDB.pickRandomSamples: numSamples has more classes than the imdb struct');
    end
    
    % Preparations: In case numSamples is a scalar, we adapt numSamples in
    % a way to preserve the label ratio
    if length(numSamples) == 1
        label_distributions = [];
        for l=1:length(imdb.meta.classes)
            label_distributions(1, l) = sum(imdb.meta.classes(l) == imdb.images.labels);
        end
        label_distributions = label_distributions ./ sum(label_distributions);
        numSamples = round(label_distributions * numSamples);
    end
    
    % Go: For each class, randomly collect numSamples(class) patches
    imdb_ = init();
    imdb_.meta = imdb.meta;
    pickedIdx = [];
    for c=1:length(numSamples)
        class_idx = find(imdb.images.labels == c);
        ridx = randperm(length(class_idx), numSamples(c));
        
        imdb_.images.data(:,:,:,end+1:end+numSamples(c)) = imdb.images.data(:,:,:, class_idx(ridx));
        imdb_.images.labels(1, end+1:end+numSamples(c)) = c;
        imdb_.images.set(1, end+1:end+numSamples(c)) = imdb.images.set(class_idx(ridx));
        if isfield(imdb.images,'filenames')
            imdb_.images.filenames(1, end+1:end+numSamples(c)) = imdb.images.filenames(class_idx(ridx));
        end
        
        % Update pickedIdx
        pickedIdx = [pickedIdx class_idx(ridx)];
            
        %if numSamples(c) < length(class_idx) % No oversampling in this case
        %    class_idx(ridx) = []; 
        %end
    end
    
    % Mean training data
    imdb_.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
    
    % Return
    imdb = imdb_;
end

