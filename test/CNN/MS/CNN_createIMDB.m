% Find and filter training & testing images in order to create an IMDB with training data
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

clear;
close all;
run ../../../Setup.m
tic;

% Options
CNN_opts;
options.data.domains = {'D'};

% Init IMDB
imdb = IMDB.init();
imdb.images.labels = zeros(0,0,0,0);
imdb.meta.classes = [options.data.classes.background options.data.classes.lesion];
if options.data.lowMemory
	[pathstr, name, ext] = fileparts(options.data.rusdbDir);
	imdb.meta.pathstr = pathstr;
	imdb.meta.name = name;
	if ~exist([imdb.meta.pathstr '/' imdb.meta.name], 'dir')
		mkdir([imdb.meta.pathstr '/' imdb.meta.name]);
	end
else
	imdb.images = rmfield(imdb.images, 'filenames');
end

% Init handles
if options.debug || options.verbose
    h1 = figure;
    h2 = figure;
    h3 = figure;
end

% Load MNIST Training images & labels
imagesA = getAllFiles(options.data.dir, '*FLAIR_preprocessed.nii.gz', true);
imagesB = getAllFiles(options.data.dir, '*T1_preprocessed.nii.gz', true);
imagesC = getAllFiles(options.data.dir, '*T2_preprocessed.nii.gz', true);
labelNIIs = getAllFiles(options.data.dir, '*Consensus.nii.gz', true);
imagesA = [imagesA; getAllFiles(options.data.dirMSKRI, '*rf2.nii.gz', true)];
imagesB = [imagesB; getAllFiles(options.data.dirMSKRI, '*rt1.nii.gz', true)];
imagesC = [imagesC; getAllFiles(options.data.dirMSKRI, '*rt2.nii.gz', true)];
labelNIIs = [labelNIIs; getAllFiles(options.data.dirMSKRI, '*s.nii.gz', true)];
train = numel(labelNIIs);
imagesA = [imagesA; getAllFiles(options.test.data.dir, '*FLAIR_preprocessed.nii.gz', true)];
imagesB = [imagesB; getAllFiles(options.test.data.dir, '*T1_preprocessed.nii.gz', true)];
imagesC = [imagesC; getAllFiles(options.test.data.dir, '*T2_preprocessed.nii.gz', true)];
labelNIIs = [labelNIIs; getAllFiles(options.test.data.dir, '*Consensus.nii.gz', true)];
imagesA = [imagesA; getAllFiles(options.test.data.dirMSKRI, '*rf2.nii.gz', true)];
imagesB = [imagesB; getAllFiles(options.test.data.dirMSKRI, '*rt1.nii.gz', true)];
imagesC = [imagesC; getAllFiles(options.test.data.dirMSKRI, '*rt2.nii.gz', true)];
labelNIIs = [labelNIIs; getAllFiles(options.test.data.dirMSKRI, '*s.nii.gz', true)];
sets = ones(1,numel(labelNIIs));
sets(train+1:end) = 2;

% Status vars
numImages = length(imagesA);
processed = 0;

% Loop over all images
for i=1:numImages
    nii_imageA = NII(imagesA{i});
    nii_imageB = NII(imagesB{i});
    nii_imageC = NII(imagesC{i});
    nii_labels = NII(labelNIIs{i});
    set = sets(i);
    
    % Some hardcoded magic to distinguish different domains and patients
    patientName = hlp_getPatientName(imagesA{i});
    patientIdx = hlp_getPatientIdx(imagesA, patientName);
    [~,domain] = hlp_getDomain(patientName);
    if domain == 'D'
        nii_imageA.permuteDimensions([1 3 2]);
        nii_imageB.permuteDimensions([1 3 2]);
        nii_imageC.permuteDimensions([1 3 2]);
        nii_labels.permuteDimensions([1 3 2]);
        nii_imageA.rot90k(3);
        nii_imageB.rot90k(3);
        nii_imageC.rot90k(3);
        nii_labels.rot90k(3);
    end
    sz = nii_imageA.size();
    if numel(options.data.domains) > 0 && numel(strmatch(domain, options.data.domains)) == 0
        continue;
    end
    if numel(options.data.patients) > 0 && sum(strcmp(options.data.patients, num2str(patientIdx))) == 0
        continue;
    end
    
    % Normalize 
    %Image.data = normalize2(Image.data, 'single');
    nii_imageA.normalize('single');
    nii_imageB.normalize('single');
    nii_imageC.normalize('single');
	
    for slice=1:sz(options.data.axis)
        % 0. Load the current image and init the labelMap
        Image = struct;
        Image.data(:,:,1) = nii_imageA.getSlice(slice, options.data.axis);
        Image.data(:,:,2) = nii_imageB.getSlice(slice, options.data.axis);
        Image.data(:,:,3) = nii_imageC.getSlice(slice, options.data.axis);
        Image.labelmap = (nii_labels.getSlice(slice, options.data.axis) > 0) + 1;
        
        % set NaN pixels to 0
        Image.data(isnan(Image.data)) = 0;
        Image.labelmap(isnan(Image.labelmap)) = 1;
        
        % Skip any slices that are totally blank
        tmp = Image.data(:,:,1);
        if length(unique(tmp(:))) == 1
           continue; 
        end
        % Skip any slices with only one label
        if length(unique(Image.labelmap(:))) == 1
           continue; 
        end
        
        if options.debug
           figure(h1), cla;
           % Meta information
           disp(['Min: ' num2str(min(Image.data(:))) ' - Max:' num2str(max(Image.data(:)))]);
           subplot(1,2,1), imagesc(Image.data); % For debugging purposes
           subplot(1,2,2), imagesc(Image.labelmap);
        end
        
        % Creating the IMDB:
        % A) Perform all rotations (rotate both image and labelmap!)
        % B) Add noise to the image
        % C) Do mirroring
        % D) Do scaling
        % E) Perhaps do random crops
    
        % Go Go Go
        for r=1:length(options.data.rotations) % Rotate the full images to speed up patch extraction
           Image.data_rotated = imrotate(Image.data, options.data.rotations(r), 'bilinear', 'crop');
           Image.labelmap_rotated = imrotate(Image.labelmap, options.data.rotations(r), 'nearest', 'crop');
           %nans = (Image.labelmap_rotated == 0);
           %Image.labelmap_rotated(find(Image.labelmap_rotated == 0)) = 1;
           %Image.data_rotated(repmat(nans, [1, 1, 3])) = 0;
           
           for s=1:length(options.data.scales)
               Image.data_scaled = imresize(Image.data_rotated, options.data.scales(s));
               Image.labelmap_scaled = imresize(Image.labelmap_rotated, options.data.scales(s), 'nearest');
               
               for is=1:length(options.data.intensityScale)
                   Image.data_iscaled = Image.data_scaled * options.data.intensityScale(is);

               for n=1:length(options.data.noise_sigmas)
                   if options.data.noise_sigmas == 0
                       Image.data_noisy = Image.data_iscaled;
                   else
                       Image.data_noisy = imnoise(Image.data_iscaled, 'gaussian', 0, options.data.noise_sigmas(n));
                   end
                   
                   if options.data.wholeSlices
                       patch = Image.data_noisy;
                       labels = Image.labelmap_scaled;
                       
                       if options.data.lowMemory
                           save([imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.mat'], 'patch', 'labels');
                           imdb.images.filenames{end+1} = [num2str(length(imdb.images.filenames)) '.mat'];
                           imdb.images.data(:,:,:,end+1) = single(0);
                           imdb.images.labels(end+1) = 0;
                       else
                           imdb.images.data(:,:,:,end+1) = patch;
                           imdb.images.labels(:,:,:,end+1) = labels;
                       end
                       imdb.images.set(end+1) = set;
                       
                       if options.data.mirror
                       % Mirror and store both the patch and the labels
                           patch = flip(patch,2);
                           labels = flip(labels,2);
                           if options.data.lowMemory
                               %imwrite(patch, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.png']);
                               %imwrite(labels, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '_labels.png']);
                               save([imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.mat'], 'patch', 'labels');
                               imdb.images.filenames{end+1} = [num2str(length(imdb.images.filenames)) '.mat'];
                               imdb.images.data(:,:,:,end+1) = single(0);
                               imdb.images.labels(end+1) = 0;
                           else
                               imdb.images.data(:,:,:,end+1) = patch;
                               imdb.images.labels(:,:,:,end+1) = labels;
                           end
                           imdb.images.set(end+1) = set;
                       end

                       if options.debug
                           figure(h2), cla;
                           subplot(1,2,1), imagesc(Image.data_noisy); % For debugging purposes
                           subplot(1,2,2), imagesc(Image.labelmap_scaled);
                       end
                   elseif options.data.randomCrops
                       rand_topleft_y = randi(size(Image.data_noisy,1) - options.data.randomCropSize(1) + 1, options.data.randomCropsPerAugmentation);
                       rand_topleft_x = randi(size(Image.data_noisy,2) - options.data.randomCropSize(2) + 1, options.data.randomCropsPerAugmentation);

                       c = 0;
                       iter = 0;
                       while c < options.data.randomCropsPerAugmentation
                           if iter == 1000
                               break;
                           end
                           iter = iter+1;
                           rand_topleft_y = randi(size(Image.data_noisy,1) - options.data.randomCropSize(1) + 1);
                           rand_topleft_x = randi(size(Image.data_noisy,2) - options.data.randomCropSize(2) + 1);
                           Image.data_randomcrop = imcrop(Image.data_noisy,[rand_topleft_x, rand_topleft_y, options.data.randomCropSize(2)-1, options.data.randomCropSize(1)-1]);
                           Image.labelmap_randomcrop = imcrop(Image.labelmap_scaled, [rand_topleft_x, rand_topleft_y, options.data.randomCropSize(2)-1, options.data.randomCropSize(1)-1]);
                           
                           % If the cropped patch does not contain lesions,
                           % retry
                           if numel(unique(Image.labelmap_randomcrop(:))) == 1
                               continue;
                           end
                           
                           patch = Image.data_randomcrop;
                           labels = Image.labelmap_randomcrop;
                           
                           % Skip any patches without brain
                           tmp = patch(:,:,1);
                           [counts,binLocations] = imhist(tmp);
                           if length(unique(tmp(:))) == 1
                               continue; 
                           end
                           if length(unique(patch(:))) == 1 %|| sum(patch(:)) < 0.25
                               continue; 
                           end
                           if length(unique(labels(:))) == 1 && unique(labels(:)) == 1
                           %    continue; 
                           end
                           if sum(counts(1:10)) > 0.5*sum(counts)
                               continue;
                           end
                           
                           % enhance labelmap: set any pixels < 0.01
                           % intensity to ignore
                           %labels(tmp < 0.001) = 0;
                           
                           if options.data.lowMemory
                               save([imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.mat'], 'patch', 'labels');
                               imdb.images.filenames{end+1} = [num2str(length(imdb.images.filenames)) '.mat'];
                               imdb.images.data(:,:,:,end+1) = single(0);
                               imdb.images.labels(end+1) = 0;
                           else
                               imdb.images.data(:,:,:,end+1) = patch;
                               imdb.images.labels(:,:,:,end+1) = labels;
                           end
                           imdb.images.set(end+1) = set;
                           
%                            if options.debug
%                                imwrite(patch, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.png']);
%                                imwrite(labels, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '_labels.png']);
%                            end

                           if options.data.mirror
                           % Mirror and store both the patch and the labels
                               patch = flip(patch,2);
                               labels = flip(labels,2);
                               if options.data.lowMemory
                                   %imwrite(patch, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.png']);
                                   %imwrite(labels, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '_labels.png']);
                                   save([imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.mat'], 'patch', 'labels');
                                   imdb.images.filenames{end+1} = [num2str(length(imdb.images.filenames)) '.mat'];
                                   imdb.images.data(:,:,:,end+1) = single(0);
                                   imdb.images.labels(end+1) = 0;
                               else
                                   imdb.images.data(:,:,:,end+1) = patch;
                                   imdb.images.labels(:,:,:,end+1) = labels;
                               end
                               imdb.images.set(end+1) = set;
                           end

                           if options.debug
                               figure(h2), cla;
                               subplot(1,2,1), imagesc(patch); % For debugging purposes
                               subplot(1,2,2), imagesc(labels);
                               drawnow;
                           end
                           c = c+1;
                       end
                   else
                       Image.labels = labelsFromBinaryMap(Image.labelmap_scaled);
                       ridx = randperm(size(Image.labels,1));
                       
                       % Now first extract patches around labeled points, possibly mirror them
                       for p=1:min(size(Image.labels,1), options.data.patchesPerAugmentation)
                           point = [Image.labels(ridx(p),1) Image.labels(ridx(p),2)];
                           try
                               patch = patchAroundPoint( Image.data_noisy, point, options.data.patchsize );
                               labels = patchAroundPoint( Image.labelmap_scaled, point, options.data.patchsize);
                           catch exc
                               continue;
                           end

                           % Skip any patches without brain
                           tmp = patch(:,:,1);
                           if length(unique(tmp(:))) == 1
                               continue; 
                           end
                           if length(unique(patch(:))) == 1 %|| sum(patch(:)) < 0.25
                               continue; 
                           end

                           if options.data.lowMemory
                               save([imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.mat'], 'patch', 'labels');
                               imdb.images.filenames{end+1} = [num2str(length(imdb.images.filenames)) '.mat'];
                               imdb.images.data(:,:,:,end+1) = single(0);
                               imdb.images.labels(end+1) = 0;
                           else
                               imdb.images.data(:,:,:,end+1) = patch;
                               imdb.images.labels(:,:,:,end+1) = labels;
                           end
                           imdb.images.set(end+1) = set;

%                            if options.debug
%                                imwrite(patch, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.png']);
%                                imwrite(labels, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '_labels.png']);
%                            end

                           if options.data.mirror
                           % Mirror and store both the patch and the labels
                               patch = flip(patch,2);
                               labels = flip(labels,2);
                               if options.data.lowMemory
                                   %imwrite(patch, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.png']);
                                   %imwrite(labels, [imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '_labels.png']);
                                   save([imdb.meta.pathstr '/' imdb.meta.name '/' num2str(length(imdb.images.filenames)) '.mat'], 'patch', 'labels');
                                   imdb.images.filenames{end+1} = [num2str(length(imdb.images.filenames)) '.mat'];
                                   imdb.images.data(:,:,:,end+1) = single(0);
                                   imdb.images.labels(end+1) = 0;
                               else
                                   imdb.images.data(:,:,:,end+1) = patch;
                                   imdb.images.labels(:,:,:,end+1) = labels;
                               end
                               imdb.images.set(end+1) = set;
                           end

                           if options.debug
                               figure(h2), cla;
                               subplot(1,2,1), imagesc(patch); % For debugging purposes
                               subplot(1,2,2), imagesc(labels);
                           end
                       end
                   end
               end
               end
           end
        end
    end
    
    % Display process
    processed = processed + 1;
    if mod(processed, floor((0.01*numImages))) == 0
        disp(['Processed: ' num2str(processed/numImages) '%']);
    end
    
    % Clearvars
    clearvars Image patch;
end

%imdb = IMDB.repartition( imdb, options.data.partitions, false );
if options.data.meanImage
    imdb.data.dataMean = mean(imdb.images.data, 4);
end

% Save RUSDB
save(options.data.imdbDir, 'imdb', '-v7.3');
toc;