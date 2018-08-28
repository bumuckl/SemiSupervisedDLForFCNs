function imdb = createIMDBData(opts)
%CREATEIMDBDATA Read uint8 RGB images and convert them into IMDB data. You
%can save separate files for each input file, or one

% Search a directory for pairs of images and csv-files. Preprocess the
% images, extract patches, label them and store them inside a IMDB struct. Can also mirror and rotate patches.
Setup;

% Variables (that you might want to change)
options = struct;
options.source_dir = '../../draft/Mitosis Data/';
options.file_extensions = '*.tif'; % The file extensions of the images
options.target_suffix = 'r5b'; % This suffix will be added to all .imdb.mat files, so you can recognize the files again
options.big_target_file = './data/mitosis_balanced_patchsize65x65_radius5.imdb.mat';
options.patchsize = [101 101]; % patchsize in [patchheight patchwidth]
options.partitions = [0.6 0.2 0.2];
options.rotations = [0 90 180 270]; % Rotation in angles
options.mirroring = 1; % 1 means true, 0 means false
options.d = 10; % Acceptable distance in px from label center for a patch to still be labeled with the respective label
options.d_whitePixels = 50; % Set to -1 if you want to ignore this. If it is positive, each circle around a label point will have only d_whitePixels pixels
options.imageScale = 1; % You can also resize the input images to speed up the process
options.low_memory = 1; % If this is true, then all the patches for a specific image will be stored in a file and deleted from workspace during patch extraction
options.verbose = 1;
options.visualize = 0;
options.n_multiplier = 1; % For each labeled patch, collect n_multiplier unlabeled patches

if nargin == 1
    options = opts;
end

% Algorithm:
% - Read all pairs of images and csv-files inside a directory, including
% subdirectories
% - Iterate over all files, stain normalize the images and extract patches
% around labeled points
% - Randomly extract the same amount of patches around points that are unlabeled
% - Create rotations and mirrored versions of each patch
% - At the end, save to target_file

% Init IMDB struct
imdb = IMDB.init();
imdb.images.data = zeros(0,0,0,0); % If this step is omitted, matlab does some weird indexing things 
imdb.meta.classes = [1 2];

% 1. Read all files inside the source directory
files = getAllFiles(options.source_dir, options.file_extensions, 1);

% 2. Iterate over all images and perform the aforementioned algorithm
%Image = struct;
for f=1:length(files)
    % Read image and image info
    Image.info = imfinfo(files{f});
    Image.data = imread(files{f});
    
    % Read corresponding CSV file and obtain labels
    [pathstr,name,ext] = fileparts(files{f});
    if ~exist([pathstr '/' name '.csv'], 'file')
        continue;
    end
    Image.labels = csvread([pathstr '/' name '.csv']); % Note: first column is y, second column is x
    
    % Extend labels with help of a labelmap with the radius specified by
    % options.d
    Image.labelmap = zeros(size(Image.data,1), size(Image.data,2));
    for p=1:size(Image.labels,1)
        if optiopns.d_whitePixels > 0
            Image.labelmap = circleAroundPoint( Image.labelmap, [Image.labels(p,1) Image.labels(p,2)], options.d, options.d_whitePixels );
        else
            Image.labelmap = circleAroundPoint( Image.labelmap, [Image.labels(p,1) Image.labels(p,2)], options.d );
        end
    end
    
    % Rescale the images if desired
    Image.data = imresize(Image.data, options.scale);
    Image.labelmap = imresize(Image.labelmap, options.scale);
    
    % Visualize patches if desired
    if options.visualize
        figure(1), cla, subplot(1,2,1), imshow(Image.data .* uint8(repmat( Image.labelmap, [1 1 3]))); % For debugging purposes
    end
    
    % Possibly mirror the image borders by patchsize/2 px in each direction
    Image.data_padded = padarray(Image.data, floor(options.patchsize ./ 2), 'symmetric');
    Image.labelmap_padded = padarray(Image.labelmap, floor(options.patchsize ./ 2)); % Pad labelmap with zeros
    
    % Normalize the staining
    [Image.data_normalized] = ColorNorm.normalizeStaining(Image.data_padded);
    
    % Go Go Go
    for r=1:length(options.rotations) % Rotate the full images to speed up patch extraction
        % Rotate image and labelmap
        Image.data_rotated = imrotate(Image.data_normalized, options.rotations(r), 'crop');
        Image.labelmap_rotated = imrotate(Image.labelmap_padded, options.rotations(r), 'crop');
        
        % Then extract label coordinates from the rotated labelmap
        Image.labels = labelsFromBinaryMap(Image.labelmap_rotated);
        
        % Now first extract patches around labeled points, possibly mirror them
        for p=1:size(Image.labels,1)
            point = [Image.labels(p,1) Image.labels(p,2)];
            patch = patchAroundPoint( Image.data_rotated, point, options.patchsize );
            imdb.images.data(:,:,:,end+1) = patch;
            if options.visualize
                figure(1), subplot(1,2,2), cla, imshow(patch); % For debugging purposes
            end
            imdb.images.labels(1,end+1) = 2;
            if options.mirroring
                % Mirror and store the patch
                imdb.images.data(:,:,:,end+1) = flipdim(patch, 2);
                imdb.images.labels(1,end+1) = 2;
            end
        end
        
        % Now collect the same amount of patches*options.n_multiplier (for instance to mimic a distribution) of unlabeled pixels
        for p=1:floor(size(Image.labels,1) * options.n_multiplier)
            point = [];
            while true
                y = randi([1 size(Image.data,1)],1,1);
                x = randi([1 size(Image.data,2)],1,1);
                point = [y x] + floor(options.patchsize ./ 2); % Correct for the padding
                if ~Image.labelmap_rotated(point) % If the point is not part of the labelmap we have a match
                    break;
                end
            end

            patch = patchAroundPoint( Image.data_rotated, point, options.patchsize );
            imdb.images.data(:,:,:,end+1) = patch;
            imdb.images.labels(1,end+1) = 1;
            if options.mirroring
                % Mirror, Rotate and store the patch
                imdb.images.data(:,:,:,end+1) = flipdim(patch, 2);
                imdb.images.labels(1,end+1) = 1;
            end  
        end
    end

    if options.verbose
       disp(['Done with patch extraction and labeling of ' files{f}]);
    end
    
    % Finally, save patches{f} and labels{f} to a file and reset them if in
    % low memory mode
    if options.low_memory
	   imdb = IMDB.repartition(imdb, options.partitions);
       save([pathstr '/' name options.target_suffix '.imdb.mat'], 'imdb', '-v7.3');
       
       % Reset imdb for next image
       clearvars imdb;
       imdb = IMDB.init();
       imdb.images.data = zeros(0,0,0,0); % If this step is omitted, matlab does some weird indexing things 
       imdb.meta.classes = [1 2];
       
       if options.verbose
            disp(['Saved to file ' pathstr '/' name options.target_suffix '.imdb.mat']);
       end
    end
end

if ~options.low_memory
    imdb = IMDB.repartition(imdb, options.partitions);
    save(options.big_target_file, 'imdb');
else
%     files = getAllFiles(options.source_dir, ['*' options.target_suffix '.imdb.mat'], 1);
%     for f=1:length(files)
%         load(files{f});
%         if f == 1
%             imdb_ = imdb;
%         else
%             imdb_.images.data(:,:,:,end+1:end+size(imdb.images.data,4)) = imdb.images.data;
%             imdb_.images.labels(1,end+1:end+size(imdb.images.labels,2)) = imdb.images.labels;
%             imdb_.images.set(1,end+1:end+size(imdb.images.set,2)) = imdb.images.set;
%         end
%         clearvars imdb;
%     end
%     imdb = imdb_;
%     clearvars imdb_;
%     imdb = IMDB.repartition(imdb, options.partitions);
%     save(options.big_target_file, 'imdb', '-v7.3');
end

end