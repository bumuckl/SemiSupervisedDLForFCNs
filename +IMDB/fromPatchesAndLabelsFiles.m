function imdb = fromPatchesAndLabelsFiles(path, set_partitions)
%FROMPATCHESANDLABELS Read all .mat files in path and convert the contained patches and labels variables returned from
%"CreatePatchesData" into an IMDB struct (i.e. used by MatConvNet)
%
% @Author: Christoph Baur
    
    % Gain access to all methods of the package
    import IMDB.*

    if nargin < 2
        set_partitions = [0.6 0.2 0.2];
    end

    imdb = init();
    
    % Read all files, iterate over all files, obtain an imdb_ struct for
    % every image and merge it with the overall imdb struct
    files = getAllFiles(path, '*.mat', 1);
    for f=1:length(files)
        load(files{f});
        tmp_patches{1} = patches{f};
        tmp_labels{1} = labels{f};
        imdb_ = fromPatchesAndLabels(tmp_patches, tmp_labels, set_partitions);
        imdb = merge(imdb, imdb_);
        clearvars tmp_patches tmp_labels imdb_
        disp(['Done with file ' files{f} '(file ' num2str(f) ' of ' num2str(length(files)) ' files)']);
    end
end

