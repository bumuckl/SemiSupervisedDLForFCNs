function imdb = init()
%INIT Returns a totally empty imdb struct. This is for convenience to make
%sure you know what this struct looks like
%
% @Author: Christoph Baur

    % Gain access to all methods of the package
    import IMDB.*

    imdb = struct;
    imdb.images = struct;
    imdb.images.data = zeros(0,0,0,0);
    imdb.images.dataMean = [];
    imdb.images.labels = [];
    imdb.images.set = [];
    imdb.images.filenames = {};
    imdb.meta.sets = {'train', 'val', 'test'};
    imdb.meta.classes = [];
end

