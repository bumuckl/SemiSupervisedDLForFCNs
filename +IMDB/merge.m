function imdb = merge( imdb_a, imdb_b )
%MERGE Merge two imdb structs
%
% @Author: Christoph Baur

    % Gain access to all methods of the package
    import IMDB.*
    
    % Make copy of imdb_a
    imdb = imdb_a;
    
    % Append imdb_b data
    imdb.images.data(:,:,:, end+1:end+size(imdb_b.images.data, 4)) = imdb_b.images.data;
    imdb.images.labels(1, end+1:end+size(imdb_b.images.labels, 2)) = imdb_b.images.labels;
    imdb.images.set(1, end+1:end+size(imdb_b.images.set, 2)) = imdb_b.images.set;
    
    % Set classes
    imdb.meta.classes = unique(imdb.images.labels);
    
    % Mean training data
    imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
end

