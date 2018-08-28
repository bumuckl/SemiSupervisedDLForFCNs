% A custom layer which takes two different input batches and constructs
% both a blob of embeddings and an associated graph adjacency matrix W. W
% is defined based on the labels of input a and b, but could in general be
% any kind of prior. TH einput here is expected to be batches of patches
% (both the data and the labels)
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

classdef EmbeddingSampler < dagnn.ElementWise
  properties
    opts = {}
    numEmbeddings = 1000 % Number of embeddings to sample
    strategy = 'fixed' % or 'distAware'
		partitions = [0.8 0.05 0.05 0.05 0.05]
    lu_ratio = 1
    debug = true
  end

  properties (Transient)
    ridx_l = []
    ridx_u = []
    h_debug = 0
    h_debug2 = 0
  end
  
  methods
    function outputs = forward(obj, inputs, params)
        global glob;
        
        data = gather(inputs{1});
        labels = gather(inputs{2});
        labels_lu = gather(inputs{3});  % Holds the actual ground-truth such that we can model a perfect prior
        [sh, sw, sc, sn] = size(data);
        
        % Reshape data
        data = reshape(permute(data, [3 1 2 4]), sc, sh*sw*sn);
        idx_l = find(~isnan(labels(:))); % indices of labeled data
        idx_u = find(isnan(labels(:))); % indices of unlabeled data
        
        % 1. Get the random samples according to the lu_ratio
        % keeping the ratio within the actual labeled data as is
        % and sampling from unlabeled data just randomly
        obj.ridx_l = idx_l(obj.getRandomEmbeddingsIdxFromLabels(labels(idx_l), obj.numEmbeddings));
        %obj.ridx_u = idx_u(randperm(numel(idx_u), min(round(numel(obj.ridx_l)*obj.lu_ratio), numel(idx_u))));
        obj.ridx_u = idx_u(obj.getRandomEmbeddingsIdxFromLabels(labels_lu(idx_u), obj.numEmbeddings*obj.lu_ratio));
        embeddings = [data(:, obj.ridx_l) data(:, obj.ridx_u)];
        labels = [labels(obj.ridx_l); labels(obj.ridx_u)];
        
        % 2. Create the perfect prior W
        labels_lu = [labels_lu(obj.ridx_l); labels_lu(obj.ridx_u)];
        W = zeros(numel(labels));
        for i=1:size(W,1)
            for j=1:i
                W(i,j) = (labels_lu(i) == labels_lu(j));
                W(j,i) = W(i,j);
            end
        end
        
        % 3. Reshape embeddings such that SemiSupervisedEmbeddingLayer can
        % handle it correctly
        embeddings = reshape(embeddings, [size(embeddings,1) 1 1 size(embeddings,2)]);
        
        % If desired, visualize debug output
        if obj.debug
            if obj.h_debug == 0
               obj.h_debug = figure;
            else
               figure(obj.h_debug), cla;
            end
            
            % Output label statistics
            classes = unique(labels_lu(:));
            for c=1:numel(classes)
               disp(['Num of embeddings with class ' num2str(classes(c)) ': ' num2str(sum(labels_lu(:) == classes(c)) / numel(labels_lu))]); 
            end
            
            labels_all = gather(inputs{3});
            ridx_channel_l = zeros(size(labels_all));
            ridx_channel_u = zeros(size(labels_all));
            ridx_channel_l(obj.ridx_l) = 1;
            ridx_channel_u(obj.ridx_u) = 1;
            for i=1:sn
                title('EmbeddingSampler Forward');
                debug_im = cat(3, labels_all(:,:,1,i)/2, ridx_channel_l(:,:,1,i), ridx_channel_u(:,:,1,i));
                subplot(sn,1,i), imagesc(debug_im);
            end
            drawnow;
            
            % Compute and output the KL Divergence between selected embeddings and the real data!
            kl_all = kldiv(embeddings, data);
            js_all = jsdiv(embeddings, data);
            glob.kl_all = [glob.kl_all kl_all];
            glob.js_all = [glob.js_all js_all];
            %kl_fwise = kldiv_featurewise(embeddings, data);
            disp(['Num Embeddings: ' num2str(size(embeddings,4))]);
            disp(['KL-Divergence (All): ' num2str(kl_all)]);
            disp(['JSL-Divergence (All): ' num2str(js_all)]);
            %disp(['KL-Divergence (Feature-wise): ' num2str(kl_fwise)]);
        end
        
        outputs = {embeddings, W, labels};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
        data = gather(inputs{1});
        labels = gather(inputs{2});
        labels_lu = gather(inputs{3});  % Holds the actual ground-truth such that we can model a perfect prior
        [sh, sw, sc, sn] = size(data);
        
        % Bring the derivatives (derOutputs) into the correct, original shape again
        % i.e. take derOutputs, split it, reshape it and put it into the
        % respective derInputs
        tmp = zeros(sc, sh*sw*sn);
        tmp(:,obj.ridx_l) = derOutputs{1}(:,:,:, 1:numel(obj.ridx_l));
        tmp(:,obj.ridx_u) = derOutputs{1}(:,:,:, numel(obj.ridx_l)+1:end);
        der = reshape(tmp, sc, sh, sw, sn);
        derInputs{1} = permute(der, [2 3 1 4]);
        
        derInputs{2} = [];
        derInputs{3} = [];
        derParams = {};
        
        % If desired, visualize debug output
        if obj.debug
            if obj.h_debug2 == 0
               obj.h_debug2 = figure;
            else
               figure(obj.h_debug2), cla;
            end
            
            for i=1:sn
                title('EmbeddingSampler Backward');
                debug_im = sum(derInputs{1}(:,:,:,i), 3) ~= 0;
                subplot(sn,1,i), imagesc(debug_im);
            end
            drawnow;
        end
    end
    
    function ridx = getRandomEmbeddingsIdxFromLabels(obj, labels, num_embeddings)
       classes = unique(labels);
       classes(classes == 0) = []; % Ignore pixels labeled with 0
       classes(isnan(classes)) = []; % Ignore pixels labeled with NaN
       
       labelsWithClass = [];
       for c=1:numel(classes)
           labelsWithClass(c) = sum(labels == classes(c));
       end
       
       ridx = [];
	   partitions_ = obj.partitions;
       for c=1:numel(classes)
           embeddings_c = find(labels == classes(c));
					 if strcmp(obj.strategy, 'distAware')
           		fac_c = numel(embeddings_c)/numel(labels); % distAware
					 else
           		fac_c = partitions_(c);
					 end
           ridx = [ridx; embeddings_c(randperm(numel(embeddings_c), round(min(num_embeddings, labelsWithClass(c))*fac_c) ))];
       end
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      xSize = inputSizes{1};
      gSize = inputSizes{2};
      outputSizes{1} = {1};
    end

    function obj = EmbeddingSampler(varargin)
      obj.load(varargin);
    end
  end
end
