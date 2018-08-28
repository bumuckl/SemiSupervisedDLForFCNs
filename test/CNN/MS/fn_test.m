%FN_TEST given a struct of options, test a model (see CNN_test.m on how to use it)
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

function [ Eval, embeddings, embeddings_labels ] = fn_test( options )

% Init handles
if options.debug || options.verbose
    h1 = figure;
    h2 = figure;
end

% Set options if they are not provided
if ~isfield(options.test, 'augmentations')
    options.test.augmentations.rotations = 0;
    options.test.augmentations.scales = 1;
end
if ~isfield(options.test, 'saveSliceVisualizations')
    options.test.saveSliceVisualizations = false;
end

% Load trained model
if strcmp('last', options.test.epoch)
    options.test.epoch = num2str(CNN.findLastCheckpoint(options.train.expDir));
    epochFile = [options.train.expDir 'net-epoch-' options.test.epoch '.mat'];
else
    epochFile = [options.train.expDir 'net-epoch-' options.test.epoch '.mat'];
end
load(epochFile, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
net.mode = 'test' ;
net.removeLayer('loss');
if isnan(net.getVarIndex('prediction'))
    net.addLayer('prob', dagnn.SoftMax(), {options.net.lastLayerBeforeLoss}, {'prediction'}, {});
end
net.rebuild();
if length(options.train.gpus) > 0
    reset(gpuDevice(1));
    net.move('gpu');
end
for l=1:length(options.test.embeddingLayers)
    net.vars(net.getVarIndex(options.test.embeddingLayers{l})).precious = true;
end

% Make eval folder if it does not exist
if ~isfield(options.test, 'evalFolderPrefix')
    options.test.evalFolderPrefix = 'eval';
end
evalFolder = [options.train.expDir options.test.evalFolderPrefix '-epoch-' options.test.epoch];
if ~exist(evalFolder, 'dir')
    mkdir(evalFolder);
end

% Init Eval struct
Eval = struct;
Eval.table = struct;
Eval.domains = {};
Eval.all = struct;

% Load MS testing Volumes
imagesA = getAllFiles(options.test.data.dir, '*FLAIR_preprocessed.nii.gz', true);
imagesB = getAllFiles(options.test.data.dir, '*T1_preprocessed.nii.gz', true);
imagesC = getAllFiles(options.test.data.dir, '*T2_preprocessed.nii.gz', true);
labelNIIs = getAllFiles(options.test.data.dir, '*Consensus.nii.gz', true);
imagesA = [imagesA; getAllFiles(options.test.data.dirMSKRI, '*rf2.nii.gz', true)];
imagesB = [imagesB; getAllFiles(options.test.data.dirMSKRI, '*rt1.nii.gz', true)];
imagesC = [imagesC; getAllFiles(options.test.data.dirMSKRI, '*rt2.nii.gz', true)];
labelNIIs = [labelNIIs; getAllFiles(options.test.data.dirMSKRI, '*s.nii.gz', true)];

% Status vars
numImages = length(imagesA);
processed = 0;

% Loop over all images
embeddings = cell(length(options.test.embeddingLayers), length(options.test.embeddingDomains));
embeddings_labels = cell(length(options.test.embeddingLayers), length(options.test.embeddingDomains));
for i=1:numImages
    nii_imageA = NII(imagesA{i});
    nii_imageB = NII(imagesB{i});
    nii_imageC = NII(imagesC{i});
    nii_labels = NII(labelNIIs{i});
    nii_responses = NII(labelNIIs{i}); % A copy of labels, but we will replace every single slice with a prediction
    
    % Some hardcoded magic to distinguish different domains and patients
    patientName = hlp_getPatientName(imagesA{i});
    [domain, domainStr] = hlp_getDomain(patientName);
    if domain == 4
        nii_imageA.permuteDimensions([1 3 2]);
        nii_imageB.permuteDimensions([1 3 2]);
        nii_imageC.permuteDimensions([1 3 2]);
        nii_labels.permuteDimensions([1 3 2]);
        nii_responses.permuteDimensions([1 3 2]);
        nii_imageA.rot90k(3);
        nii_imageB.rot90k(3);
        nii_imageC.rot90k(3);
        nii_labels.rot90k(3);
        nii_responses.rot90k(3);
    end
    sz = nii_imageA.size();
    patientIdx = hlp_getPatientIdx(imagesA, patientName);
    if numel(options.test.domains) > 0 && numel(strmatch(domainStr, options.test.domains)) == 0
        continue;
    end
    if numel(options.test.patients) > 0 && sum(strcmp(options.test.patients, num2str(patientIdx))) == 0
        continue;
    end
    
    % Normalize 
    nii_imageA.normalize('single');
    nii_imageB.normalize('single');
    nii_imageC.normalize('single');
	
    if exist([evalFolder '/' patientName '.nii'], 'file')
        nii_responses = NII([evalFolder '/' patientName '.nii']);
    else
        for slice=1:sz(options.data.axis)
            % 0. Load the current image and init the labelMap
            Image = struct;
            Image.data(:,:,1) = nii_imageA.getSlice(slice, options.data.axis);
            Image.data(:,:,2) = nii_imageB.getSlice(slice, options.data.axis);
            Image.data(:,:,3) = nii_imageC.getSlice(slice, options.data.axis);
            Image.labelmap = (nii_labels.getSlice(slice, options.data.axis) > 0); % Already binary, i.e. 0 or 1
            
            % Skip any slices that are totally blank
%             tmp = Image.data(:,:,1);
%             if length(unique(tmp(:))) == 1 && unique(Image.labelmap(:)) == 1
%                nii_responses.setSlice(options.data.axis, slice, zeros(size(Image.labelmap)));
%                continue; 
%             end
            
            tmp_responses = zeros(size(Image.labelmap,1), size(Image.labelmap,2), 0);
            for r=1:length(options.test.augmentations.rotations)
                Image.data_rotated = imrotate(Image.data, options.test.augmentations.rotations(r), 'bilinear', 'crop');
           
                for s=1:length(options.test.augmentations.scales)
                    Image.data_scaled = imresize(Image.data_rotated, options.test.augmentations.scales(s));
                    
                    % Now there comes a trick...
                    % U-Net, thx to pooling & deconvolution doe snot always
                    % output the right size. We take care of that by
                    % adjusting the input appropriately
                    padded_y = false;
                    padded_x = false;
                    if rem(size(Image.data_scaled,1)/8,1) ~= 0
                        old_height = size(Image.data_scaled,1);
                        new_height = ceil(old_height/8)*8; 
                        Image.data_scaled(end:end+(new_height - old_height), :, :) = 0;
                        padded_y = true;
                    end
                    if rem(size(Image.data_scaled,2)/8,1) ~= 0
                        old_width = size(Image.data_scaled,2);
                        new_width = ceil(old_width/8)*8; 
                        Image.data_scaled(:, end:end+(new_width - old_width), :) = 0;
                        padded_x = true;
                    end
                    %Image.data_scaled(end:end+3,:,:) = 0; 
                
                    % Prepare input slice and predict
                    inputs = {'input', single(reshape(Image.data_scaled, [size(Image.data_scaled,1), size(Image.data_scaled,2), size(Image.data_scaled,3), 1]))} ;
                    if length(options.train.gpus) > 0
                        inputs{2} = gpuArray(inputs{2});
                    end
                    net.eval(inputs);

                    % Gather prediction
                    response = gather(net.vars(net.getVarIndex('prediction')).value);
                    % hard coded! 
                    if padded_y
                        response(end-(new_height - old_height -1):end,:,:) = []; 
                    end
                    if padded_x
                        response(:, end-(new_width - old_width -1):end,:) = [];
                    end
                    response_upsampled = imresize(squeeze(response(:,:,2)), [sz(1) sz(2)]);
                    response_inv_rotated = imrotate(response_upsampled, -options.test.augmentations.rotations(r), 'bilinear', 'crop');
                    
                    % Add to tmp_responses
                    response_inv_rotated(find(response_inv_rotated < 0)) = 0;
                    tmp_responses(:,:,end+1) = response_inv_rotated;
                end
            end
            consensus_response = sum(tmp_responses,3) ./ size(tmp_responses, 3);
            consensus_response2 = geomean(tmp_responses, 3);
            %consensus_response_binary = consensus_response2 > 0.99;
            
%             mask_idx = find(Image.data(:,:,1) < 0.1);
%             consensus_response(mask_idx) = 0;
%             consensus_response2(mask_idx) = 0;
            consensus_response_binary = consensus_response2 > 0.99;
            
            % Visualize
            if options.debug
                figure(10);
                for m=1:size(tmp_responses,3)
                    subplot(1,size(tmp_responses,3)+3,m), imagesc(tmp_responses(:,:,m));
                end
                subplot(1,size(tmp_responses,3)+3,m+1), imagesc(consensus_response);
                subplot(1,size(tmp_responses,3)+3,m+2), imagesc(consensus_response2);
                subplot(1,size(tmp_responses,3)+3,m+3), imagesc(Image.labelmap);
                drawnow;
            end
            
            % If desired, save the curent slice visualization
            % - the image shows the actual MR slice
            % - TP are marked in green
            % - FP are marked in orange
            % - FN are marked in red
            if options.test.saveSliceVisualizations
                sliceVisuFolder = [evalFolder '/' patientName];
                if ~exist(sliceVisuFolder, 'dir')
                    mkdir(sliceVisuFolder);
                end
                
                tmp_slice = repmat(Image.data(:,:,1), [1 1 3]);
                tmp_sliceVisuR = Image.data(:,:,1); % Fill all channels with the FLAIR MR slice
                tmp_sliceVisuG = Image.data(:,:,1); % Fill all channels with the FLAIR MR slice
                tmp_sliceVisuB = Image.data(:,:,1); % Fill all channels with the FLAIR MR slice
                TP_mask = (consensus_response_binary & Image.labelmap);
                FP_mask = (consensus_response_binary & ~Image.labelmap);
                FN_mask = (~consensus_response_binary & Image.labelmap);
                tmp_sliceVisuR(TP_mask) = 0;
                tmp_sliceVisuG(TP_mask) = 1;
                tmp_sliceVisuB(TP_mask) = 0;
                tmp_sliceVisuR(FP_mask) = 1;
                tmp_sliceVisuG(FP_mask) = 0.65;
                tmp_sliceVisuB(FP_mask) = 0;
                tmp_sliceVisuR(FN_mask) = 1;
                tmp_sliceVisuG(FN_mask) = 0;
                tmp_sliceVisuB(FN_mask) = 0;
                tmp_slice(:,:,1) = tmp_sliceVisuR;
                tmp_slice(:,:,2) = tmp_sliceVisuG;
                tmp_slice(:,:,3) = tmp_sliceVisuB;
                imwrite(tmp_slice, [sliceVisuFolder '/' num2str(slice) '.png']);
            end
            
            % Set slice of predicted volume
            nii_responses.setSlice(options.data.axis, slice, consensus_response);
            
            % Perhaps gather embeddings
            if options.test.embeddings || options.test.pca || options.test.tsne
                for l=1:length(options.test.embeddingLayers)
                    % Skip any domain not part of
                    % options.test.embeddingDomains (if it is not empty)
                    if isfield(options.test, 'embeddingDomains')
                        if numel(options.test.embeddingDomains) > 0 && numel(strmatch(domainStr, options.test.embeddingDomains)) == 0
                            continue;
                        end
                    end
                    
                    intermediate = gather(net.vars(net.getVarIndex(options.test.embeddingLayers{l})).value);
                    [she, swe, sce, sne] = size(intermediate);
                    intermediate = reshape(permute(intermediate, [3 1 2 4]), sce, she*swe*sne);
                    intermediate_labels = imresize(Image.labelmap, [she swe], 'nearest');
                    intermediate_labels = intermediate_labels(:);
%                     numEmbeddingsPerSlice = 50;
%                     if numEmbeddingsPerSlice > 0
%                         % Per slice, extract a limited number of embeddings due
%                         % to memory limitations
%                         classes = unique(intermediate_labels);
%                         labelsWithClass = [];
%                         for c=1:numel(classes)
%                             labelsWithClass(c) = sum(intermediate_labels == classes(c));
%                         end
%                         % Choose ridx in a way such that classes are balanced
%                         ridx = [];
%                         for c=1:numel(classes)
%                             embeddings_c = find(intermediate_labels == classes(c));
%                             ridx = [ridx; embeddings_c(randperm(numel(embeddings_c), min((1/numel(classes))*numEmbeddingsPerSlice, min(labelsWithClass) )))];
%                         end
%                     else
%                        ridx = 1:size(intermediate,2);
%                     end
%                     stepsizes = [500, 100];
%                     classes = unique(intermediate_labels);
%                     ridx = [];
%                     for c=1:numel(classes)
%                         labelsWithClass = find(intermediate_labels == classes(c));
%                         ridx = [ridx; labelsWithClass(1:stepsizes(c):numel(labelsWithClass))];
%                     end
                    ridx = fn_getTestEmbeddingIdx( patientName, intermediate_labels, 20, slice );
                    embeddings{l,domain} = [embeddings{l} intermediate(:, ridx)];
                    embeddings_labels{l,domain} = [embeddings_labels{l}; intermediate_labels(ridx)];
                end
            end

            if options.debug
                figure(h1), cla;
                subplot(1,3,1), imagesc(Image.data);
                subplot(1,3,2), imagesc(Image.labelmap);
                subplot(1,3,3), imagesc(consensus_response);
                %pause;
            end

        end
    end
    
    Eval.table(i).patient = patientName;
    Eval.table(i).domain = domain;
    Eval.table(i).groundtruth = nii_labels.getData() > 0;
    Eval.table(i).prediction = nii_responses.getData() > 0.99;
    Eval.table(i).TP = sum(Eval.table(i).groundtruth(:) .* Eval.table(i).prediction(:));
    Eval.table(i).TN = sum((1-Eval.table(i).groundtruth(:)) .* (1-Eval.table(i).prediction(:)));
    Eval.table(i).FP = sum((1-Eval.table(i).groundtruth(:)) .* Eval.table(i).prediction(:));
    Eval.table(i).FN = sum(Eval.table(i).groundtruth(:) .* (1-Eval.table(i).prediction(:)));
    Eval.table(i).Precision = Eval.table(i).TP / (Eval.table(i).TP + Eval.table(i).FP);
    Eval.table(i).Recall = Eval.table(i).TP / (Eval.table(i).TP + Eval.table(i).FN);
    Eval.table(i).Fscore = 2*Eval.table(i).Precision*Eval.table(i).Recall/(Eval.table(i).Precision + Eval.table(i).Recall);
    Eval.table(i).fscore = 2*(sum(Eval.table(i).prediction(:) .* Eval.table(i).groundtruth(:))) / (sum(Eval.table(i).prediction(:)) + sum(Eval.table(i).groundtruth(:)));
    Eval.table(i).error = sum(Eval.table(i).prediction(:) ~= Eval.table(i).groundtruth(:)) / numel(Eval.table(i).prediction(:));
    
    % If desired, save the predicted volume
    if options.test.savePredictedVolumes && ~exist([evalFolder '/' patientName '.nii'], 'file')
        nii_responses.save([evalFolder '/' patientName '.nii']);
    end
    if options.test.savePredictedBinaryVolumes && ~exist([evalFolder '/' patientName '.binary.nii'], 'file')
        nii_responses.threshold(0.5);
        nii_responses.save([evalFolder '/' patientName '.binary.nii']);
    end
end

% ROC curve
if options.test.plotROC
    tmp_gt = [];
    tmp_responses = [];
    for i=1:size(Eval.table,2)
        tmp_gt = [tmp_gt; Eval.table(i).groundtruth(:)];
        tmp_responses = [tmp_responses; Eval.table(i).responses(:)];
    end
    figure, plotroc(tmp_gt, tmp_responses);
end

% t-SNE
if options.test.tsne
	% perform t-SNE on the feature embeddings to see how they cluster
	for l=1:length(options.test.embeddingLayers)
		if exist([evalFolder '/embedding_' options.test.embeddingLayers{l} '.pdf'], 'dir')
			continue;
		end
		h = figure;
		mappedX = tsne(embeddings{l}', [], 2, 30, 10);
        gscatter(mappedX(:,1), mappedX(:,2), embeddings_labels{l});
        if options.test.savefig
           print(h, [evalFolder '/embedding_' options.test.embeddingLayers{l} '.pdf'], '-dpdf'); 
           print(h, [evalFolder '/embedding_' options.test.embeddingLayers{l} '.png'], '-dpng');
        end
        close(h);
	end
end

% PCA
if options.test.pca
	% perform t-SNE on the feature embeddings to see how they cluster
	for l=1:length(options.test.embeddingLayers)
		if exist([evalFolder '/embedding_pca_' options.test.embeddingLayers{l} '.pdf'], 'dir')
			continue;
		end
		h = figure;
		[coeff, score] = pca(embeddings{l}');
        gscatter(score(:,1), score(:,2), embeddings_labels{l});
        if options.test.savefig
           print(h, [evalFolder '/embedding_pca_' options.test.embeddingLayers{l} '.pdf'], '-dpdf');
           print(h, [evalFolder '/embedding_pca_' options.test.embeddingLayers{l} '.png'], '-dpng');
        end
        close(h);
	end
end

% PCA 3D
if options.test.pca3D
	% perform t-SNE on the feature embeddings to see how they cluster
	for l=1:length(options.test.embeddingLayers)
		if exist([evalFolder '/embedding_pca3D_' options.test.embeddingLayers{l} '.pdf'], 'dir')
			continue;
		end
		h = figure;
		colors = brewermap(length(options.test.embeddingDomains)*2,'Set1'); 
		[coeff, score] = pca(embeddings{l}');
		for d=1:length(options.test.embeddingDomains)
        bg_idx = find(embeddings_labels{l} == 1);
        fg_idx = find(embeddings_labels{l} == 2);
        scatter3(score(bg_idx,1), score(bg_idx,2), score(bg_idx,3), 36, colors(d*2-1)), hold on;
        scatter3(score(fg_idx,1), score(fg_idx,2), score(fg_idx,3), 36, colors(d*2)), drawnow;
		end
		legend(options.test.embeddingLayers)
    if options.test.savefig
        savefig([evalFolder '/embedding_pca3D_' options.test.embeddingLayers{l} '.fig']);
       print(h, [evalFolder '/embedding_pca3D_' options.test.embeddingLayers{l} '.pdf'], '-dpdf');
       print(h, [evalFolder '/embedding_pca3D_' options.test.embeddingLayers{l} '.png'], '-dpng');
    end
    close(h);
	end
end

%% Compute errors

% Per Domain & Overall
alldomains = [];
for r=1:size(Eval.table,2)
    alldomains = [alldomains Eval.table(r).domain];
end
domains = unique(alldomains);
Eval.all.avgfscore = 0;
Eval.all.avgerror = 0;
for d=1:numel(domains)
    domain_idx = find(alldomains == domains(d));
    Eval.domains(d).avgfscore = 0;
    Eval.domains(d).avgerror = 0;
    for i=1:numel(domain_idx)
        Eval.domains(d).avgfscore = Eval.domains(d).avgfscore + Eval.table(domain_idx(i)).fscore;
        Eval.domains(d).avgerror = Eval.domains(d).avgerror + Eval.table(domain_idx(i)).error;
    end
    Eval.all.avgfscore = Eval.all.avgfscore + Eval.domains(d).avgfscore;
    Eval.all.avgerror = Eval.all.avgerror + Eval.domains(d).avgerror;
    Eval.domains(d).domain = domains(d);
    Eval.domains(d).avgfscore = Eval.domains(d).avgfscore / numel(domain_idx);
    Eval.domains(d).avgerror = Eval.domains(d).avgerror / numel(domain_idx); 
end
Eval.all.avgfscore = Eval.all.avgfscore / size(Eval.table,2);
Eval.all.avgerror = Eval.all.avgerror / size(Eval.table,2);

% Compute cluster similarity
if options.test.embeddings
    Eval.clusterQualityDaviesBouldin = {};
    Eval.clusterQualitySilhouette = {};
    for l=1:length(options.test.embeddingLayers)
        Eval.clusterQualityDaviesBouldin{l} = evalclusters(embeddings{l}', embeddings_labels{l}+1, 'DaviesBouldin');
        Eval.clusterQualitySilhouette{l} = evalclusters(embeddings{l}', embeddings_labels{l}+1, 'Silhouette');
    end
end

if options.test.savefig
    save([evalFolder '/Eval.mat'], 'Eval'); 
    T = struct2table(Eval.table);
    T.prediction = [];
    T.groundtruth =[];
    writetable(T,[evalFolder '/table.csv']);
end
	
if options.debug || options.verbose
	 close(h1);
	 close(h2);
end

end
