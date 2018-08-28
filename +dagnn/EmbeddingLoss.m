% EMBEDDINGLOSS for supervised embedding
%
% Author: Christoph Baur
classdef EmbeddingLoss < dagnn.ElementWise
  properties
    opts = {}
	lambda = 0.01
	margin = 10
    numEmbeddings = 1000 % The number of random feature embeddings to extract if labels are also images
    loss = 'cbaur'
    debug = true
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    ridx = []
    D = []
    N = []
    h_debug = 0
    h_debug2 = 0
    h_debug3 = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      [sh sw sc sn] = size(inputs{1});
	  labels = gather(inputs{2});
	  % Case 1: there is exactly one label per data sample n out of sn samples
      if numel(labels) == sn
        embeddings = reshape(gather(inputs{1}), sh*sw*sc, sn);
      % Case 2: there is a label image for every sample n out of sn samples
	  % We do random feature embedding
      else
        embeddings = reshape(permute(gather(inputs{1}), [3 1 2 4]), sc, sh*sw*sn);
        labels = labels(:); % Make sure that labels and input have the same resolution!
        if obj.numEmbeddings > 0
           classes = unique(labels);
           classes(find(classes == 0)) = []; % Ignore pixels labeled with 0
           labelsWithClass = [];
           for c=1:numel(classes)
               labelsWithClass(c) = sum(labels == classes(c));
           end
           obj.ridx = [];
           % Choose ridx in a way such that classes are balanced
%            for c=1:numel(classes)
%                embeddings_c = find(labels == classes(c));
%                obj.ridx = [obj.ridx; embeddings_c(randperm(numel(embeddings_c), min((1/numel(classes))*obj.numEmbeddings, min(labelsWithClass) )))];
%            end
           % Choose ridx in a way such that classes are distributed as the
           % labels
           partition = [0.8 0.2];
           for c=1:numel(classes)
               embeddings_c = find(labels == classes(c));
               %fac_c = numel(embeddings_c)/numel(labels);
               fac_c = partition(c);
               obj.ridx = [obj.ridx; embeddings_c(randperm(numel(embeddings_c), round(min(obj.numEmbeddings, labelsWithClass(c))*fac_c) ))];
           end
           embeddings = embeddings(:, obj.ridx);
           labels = labels(obj.ridx);
        end
      end
	  
      if strcmp(obj.loss, 'cbaur')
        outputs{1} = obj.cbaur_fwd(embeddings, labels);
      elseif strcmp(obj.loss, 'anomaly')
        outputs{1} = obj.anomaly_fwd(embeddings, labels);
	  elseif strcmp(obj.loss, 'anomaly_normalized')
        outputs{1} = obj.anomaly_normalized_fwd(embeddings, labels);
      elseif strcmp(obj.loss, 'hadsell')
        outputs{1} = obj.dirLIM_fwd(embeddings, labels);
      elseif strcmp(obj.loss, 'hadsell_norm')
        outputs{1} = obj.dirLIMNormalized_fwd(embeddings, labels);
      elseif strcmp(obj.loss, 'cosine')
        outputs{1} = obj.cosine_fwd(embeddings, labels);
      end
	  
      n = obj.numAveraged ;
      m = n + size(embeddings, 2) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
      
      if obj.debug
        if obj.h_debug == 0
          obj.h_debug = figure;
        else
          figure(obj.h_debug), cla;
        end
        subplot(1,2,1), imagesc(embeddings);
        subplot(1,2,2), imagesc(labels);
        drawnow;
        
        if obj.numEmbeddings > 0
            if obj.h_debug2 == 0
               obj.h_debug2 = figure;
            else
               figure(obj.h_debug2), cla;
            end

            labels = gather(inputs{2});
            ridx_channel = zeros(size(labels));
            ridx_channel(obj.ridx) = 1;
            for i=1:sn
                title('EmbeddingLoss Forward');
                debug_im = cat(3, labels(:,:,1,i)/2, ridx_channel(:,:,1,i), ridx_channel(:,:,1,i));
                subplot(sn,1,i), imagesc(debug_im);
            end
            drawnow;
        end
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
	  [sh sw sc sn] = size(inputs{1});
	  labels = gather(inputs{2});
      
      % Case 1: there is exactly one label per data sample n out of sn samples
      if numel(labels) == sn
        embeddings = reshape(gather(inputs{1}), sh*sw*sc, sn);
      % Case 2: there is a label image for every sample n out of sn samples
	  % We do random feature embedding
      else
        embeddings = reshape(permute(gather(inputs{1}), [3 1 2 4]), sc, sh*sw*sn);
        labels = labels(:); % Make sure that labels and prediction have same resolution!
        if obj.numEmbeddings > 0
           embeddings = embeddings(:, obj.ridx);
           labels = labels(obj.ridx);
        end
      end
	  
      if strcmp(obj.loss, 'cbaur')
        der = obj.cbaur_bwd(embeddings, labels);
      elseif strcmp(obj.loss, 'anomaly')
        der = obj.anomaly_bwd(embeddings, labels);
	  elseif strcmp(obj.loss, 'anomaly_normalized')
        der = obj.anomaly_normalized_bwd(embeddings, labels);
      elseif strcmp(obj.loss, 'hadsell')
        der = obj.dirLIM_bwd(embeddings, labels);
      elseif strcmp(obj.loss, 'hadsell_norm')
        der = obj.dirLIMNormalized_bwd(embeddings, labels);
      elseif strcmp(obj.loss, 'cosine')
        der = obj.cosine_bwd(embeddings, labels);
      end
      
      % Case 1
      if numel(labels) == sn
          derInputs{1} = reshape(der, sh, sw, sc, sn);
      % Case 2 - Continued...
      else
          tmp = der;
          if obj.numEmbeddings > 0
             tmp = zeros(sc, sh*sw*sn);
             tmp(:,obj.ridx) = der;
          end
          der = reshape(tmp, sc, sh, sw, sn);
          derInputs{1} = permute(der, [2 3 1 4]);
          
          % If desired, visualize debug output
          if obj.debug
            if obj.h_debug3 == 0
               obj.h_debug3 = figure;
            else
               figure(obj.h_debug3), cla;
            end

            for i=1:sn
                title('EmbeddingLoss Backward');
                debug_im = sum(derInputs{1}(:,:,:,i), 3) ~= 0;
                subplot(sn,1,i), imagesc(debug_im);
            end
            drawnow;
          end
      end
	
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = EmbeddingLoss(varargin)
      obj.load(varargin) ;
    end
    
    % 
	% Different embedding schemes
	%
	
	% My own trial & error stuff
	function out = cbaur_fwd(obj, embeddings, labels)
	  W = zeros(size(embeddings,2)); % build weight matrix online
	  D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:size(embeddings,2)
		  D(i,j) = sum( (embeddings(:,i) - embeddings(:,j)).^2 );
		  if labels(i) == labels(j)
			W(i,j) = 1;
		  else
			W(i,j) = 0;
		  end
		end
      end
	  
      D = D ./ size(embeddings,2);
	  out = obj.lambda * sum(sum((W .* D)));
    end
    
	function derOut = cbaur_bwd(obj, embeddings, labels)
      W = zeros(size(embeddings,2)); % build weight matrix online
      N = size(embeddings,1);
	  D = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:size(embeddings,2)
          if labels(i) == labels(j)
			W(i,j) = 1;
		  else
			W(i,j) = 0;
          end
          D(:,i) = D(:,i) + W(i,j)*(embeddings(:,i) - embeddings(:,j));
		end
	  end
	  
	  derOut = obj.lambda * D ./ size(embeddings,2); % No need to divide by N because cnn_train_dagn does it explicitly
    end
    
    % Anomaly - My own trial & error stuff - A special version of Hadsell
    % et al.
	function out = anomaly_fwd(obj, embeddings, labels)
	  D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:size(embeddings,2)
		  if labels(i) == labels(j) && labels(i) == 1 && labels(j) == 1 % Enforce small distances among feature representations of background
			D(i,j) = sum( (embeddings(:,i) - embeddings(:,j)).^2 );
          else
            if labels(i) ~= labels(j)
              D(i,j) = max(0, obj.margin - sum( (embeddings(:,i) - embeddings(:,j)).^2 ) );
            end
		  end
		end
      end
	  
      D = D ./ size(embeddings,2);
	  out = obj.lambda * sum(sum((D)));
    end
    
	function derOut = anomaly_bwd(obj, embeddings, labels)
	  D = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:size(embeddings,2)
          if labels(i) == labels(j) && labels(i) == 1 && labels(j) == 1
			D(:,i) = D(:,i) + 2*(embeddings(:,i) - embeddings(:,j));
		  else
			if labels(i) ~= labels(j) && (obj.margin - sum( (embeddings(:,i) - embeddings(:,j)).^2 ) >= 0)
			  D(:,i) = D(:,i) - 2*(embeddings(:,i) - embeddings(:,j));
			end
          end
		end
	  end
	  
	  derOut = obj.lambda * D ./ size(embeddings,2); % No need to divide by N because cnn_train_dagn does it explicitly
	end
	
	function out = anomaly_normalized_fwd(obj, embeddings, labels)
	  % Normalize embeddings first
	  embeddings = embeddings ./ repmat( sqrt(sum(embeddings.^2,1)), size(embeddings,1), 1 );
	
	  D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:size(embeddings,2)
		  if labels(i) == labels(j) && labels(i) == 1 && labels(j) == 1 % Enforce small distances among feature representations of background
			D(i,j) = sum( (embeddings(:,i) - embeddings(:,j)).^2 );
          else
            if labels(i) ~= labels(j)
              D(i,j) = max(0, obj.margin - sum( (embeddings(:,i) - embeddings(:,j)).^2 ) );
            end
		  end
		end
      end
	  
      D = D ./ size(embeddings,2);
	  out = obj.lambda * sum(sum((D)));
    end
    
	function derOut = anomaly_normalized_bwd(obj, embeddings, labels)
	  % Normalize embeddings first
	  N = ( sqrt(sum(embeddings.^2,1)) );
	  embeddings = embeddings ./ repmat(N, size(embeddings,1), 1);
	  
	  D = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:size(embeddings,2)
          if labels(i) == labels(j) && labels(i) == 1 && labels(j) == 1
			D(:,i) = D(:,i) + 2*(embeddings(:,i) - embeddings(:,j))*N(i);
		  else
			if labels(i) ~= labels(j) && (obj.margin - sum( (embeddings(:,i) - embeddings(:,j)).^2 ) >= 0)
			  D(:,i) = D(:,i) - 2*(embeddings(:,i) - embeddings(:,j))*N(i);
			end
          end
		end
	  end
	  
	  derOut = obj.lambda * D ./ size(embeddings,2); % No need to divide by N because cnn_train_dagn does it explicitly
	end
	
	% What about a tukey regularized loss?
	
	% Hadsell et al. 2006
	function out = dirLIM_fwd(obj, embeddings, labels)
	  obj.D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:i
		  if labels(i) == labels(j)
			obj.D(i,j) = sum( (embeddings(:,i) - embeddings(:,j)).^2 );
            obj.D(j,i) = obj.D(i,j);
          else
			obj.D(i,j) = max(0, obj.margin - sum( (embeddings(:,i) - embeddings(:,j)).^2 ) );
            obj.D(j,i) = obj.D(i,j);
		  end
		end
      end
	  
	  out = sum(sum((obj.D ./ size(embeddings,2))));
	end
	
	function derOut = dirLIM_bwd(obj, embeddings, labels)
	  Der = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:i
          tmp = 2*(embeddings(:,i) - embeddings(:,j));
          if obj.D(i,j) > 0
             tmp = tmp / obj.D(i,j); 
          end
          if labels(i) == labels(j)
			Der(:,i) = Der(:,i) + tmp;
            Der(:,j) = Der(:,j) - tmp;
		  else
			if obj.margin - obj.D(i,j) >= 0
			  Der(:,i) = Der(:,i) - tmp;
              Der(:,j) = Der(:,j) + tmp;
			end
          end
		end
      end
	  
      Der = Der ./ size(embeddings,2); % Average sum of all embeddings
      derOut = obj.lambda * Der;
    end
    
    function out = dirLIMNormalized_fwd(obj, embeddings, labels)
      % Normalize embeddings first
	  obj.N = ( sqrt(sum(embeddings.^2,1)) );
	  embeddings = embeddings ./ repmat(obj.N, size(embeddings,1), 1);
      
	  obj.D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:i
		  if labels(i) == labels(j)
			obj.D(i,j) = sum( (embeddings(:,i) - embeddings(:,j)).^2 );
            obj.D(j,i) = obj.D(i,j);
          else
			obj.D(i,j) = max(0, obj.margin - sum( (embeddings(:,i) - embeddings(:,j)).^2 ) );
            obj.D(j,i) = obj.D(i,j);
		  end
		end
      end
      obj.D = obj.D ./ size(embeddings,2);
	  
	  out = sum(sum((obj.D)));
	end
	
	function derOut = dirLIMNormalized_bwd(obj, embeddings, labels)
      % Normalize embeddings first
      embeddings_orig = embeddings;
	  embeddings = embeddings ./ repmat(obj.N, size(embeddings,1), 1);
      
	  Der = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:i
          tmp = 2*(embeddings(:,i) - embeddings(:,j)) .* (obj.N(i) - embeddings(:,i).*embeddings_orig(:,i)) ./ (obj.N(i)^2);
          if labels(i) == labels(j)
			Der(:,i) = Der(:,i) + tmp;
            Der(:,j) = Der(:,j) - tmp;
		  else
			if obj.margin - obj.D(i,j) >= 0
			  Der(:,i) = Der(:,i) - tmp;
              Der(:,j) = Der(:,j) + tmp;
			end
          end
		end
      end
	  
      Der = Der ./ size(embeddings,2); % Average sum of all embeddings *OLD*
      derOut = obj.lambda * Der;
    end
    
    function out = cosine_fwd(obj, embeddings, labels)
	  obj.D = zeros(size(embeddings,2));
      fac = 1/size(embeddings,2);
	  for i=1:size(embeddings,2)
		for j=1:i
          absum = sum( embeddings(:,i) .* embeddings(:,j) );
          a2sumb2sum = sum( embeddings(:,i).^2 ) * sum( embeddings(:,i).^2 );
          distance = fac * (1 - absum / sqrt(a2sumb2sum));
		  if labels(i) == labels(j)
			obj.D(i,j) = distance;
          else
			obj.D(i,j) = max(0, obj.margin - distance);
          end
          obj.D(j,i) = obj.D(i,j);
		end
      end
      
      out = sum(obj.D(:));
    end
    
    function derOut = cosine_bwd(obj, embeddings, labels)      
      Der = zeros(size(embeddings));
      fac = 1/size(embeddings,2);
	  for i=1:size(embeddings,2)
		for j=1:i
          a = embeddings(:,i);
          b = embeddings(:,j);
          a2sum = sum( a.^2 );
          b2sum = sum( b.^2 );
          sqrta2sumb2sum = sqrt(a2sum*b2sum);
		  if labels(i) == labels(j)
			Der(:,i) = Der(:,i) - fac*(b./sqrta2sumb2sum - obj.D(i,j) * a./a2sum);
            Der(:,j) = Der(:,j) - fac*(a./sqrta2sumb2sum - obj.D(i,j) * b./b2sum);
          else
			if obj.margin - obj.D(i,j) >= 0
			  Der(:,i) = Der(:,i) + fac*(b./sqrta2sumb2sum - obj.D(i,j) * a./a2sum);
              Der(:,j) = Der(:,j) + fac*(a./sqrta2sumb2sum - obj.D(i,j) * b./b2sum);
			end
          end
		end
      end
	  
	  derOut = obj.lambda .* Der;
    end
	
	% LapSVM
	function out = lapSVM_fwd(obj, embeddings, labels)
	
	end
	function derOut = lapSVM_bwd(obj, embeddings, labels)
	
	end
	
  end
end
