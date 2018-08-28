function savePatches( imdb, path )
%SAVEPATCHES

    if isfield(imdb.images,'filenames') && length(imdb.images.filenames) > 0
        numPatches = length(imdb.images.filenames);
        for i=1:numPatches
           file = imread([imdb.meta.name '/' imdb.images.filenames{i}]);
           imwrite(file, [path '/' num2str(i) '.png']);
        end
    else
        numPatches = size(imdb.images.data, 4);
        for i=1:numPatches
           file = squeeze(imdb.images.data(:,:,:,i));
           imwrite(file, [path '/' num2str(i) '.png']);
        end
    end
end

