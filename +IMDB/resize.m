function imdb = resize( imdb, size_or_scale )
%REPARTITION Batch resize all the images inside the provided imdbstruct
%using either a vector of target dimensions or a scale.
%
% INPUT:
%
%   imdb = the imdb struct that you want to repartition
%   size_or_scale = a 2 element vector of target dimensions or a scalar for
%   resizing
%
% @Author: Christoph Baur

    % Gain access to all methods of the package
    import IMDB.*
    
    data = imdb.images.data;
    imdb.images.data =  zeros(0,0,0,0);
    for i=1:size(data,4)
        imdb.images.data(:,:,:,i) = imresize(data(:,:,:,i), size_or_scale);
    end
end

