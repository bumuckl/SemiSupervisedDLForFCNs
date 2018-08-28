% SEMISUPERVISEDEMBEDDINGLOSS for semi-supervised graph-embedding
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
classdef SemiSupervisedEmbeddingLoss < dagnn.ElementWise
  properties
    opts = {}
    loss = 'hadsell'
	lambda = 0.0001
	margin = 10
    numEmbeddings = 1000
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    D = []
    N = []
    cossim = []
    epsilon = 0.000000000001
  end

  methods
    function outputs = forward(obj, inputs, params)
      W = gather(inputs{3}); % W is a square matrix which contains graph distances between all the patches in the current batch
      labels = gather(inputs{2});
      [sh, sw, sc, sn] = size(inputs{1});
            
      % Case 1: there is exactly one label per data sample n out of sn samples
      embeddings = reshape(gather(inputs{1}), sh*sw*sc, sn);
%       if strcmp(obj.net.mode, 'test') % Remove unlabeled items because it kills validation
%         embeddings = embeddings(:,~isnan(labels));
%         W = W(:,~isnan(labels));
%         W = W(~isnan(labels),:);
%       end
      
      if strcmp(obj.loss, 'hadsell')
        outputs{1} = obj.dirLIM_fwd(embeddings, W);
      elseif strcmp(obj.loss, 'hadsell_norm')
        outputs{1} = obj.dirLIMNormalized_fwd(embeddings, W);
      elseif strcmp(obj.loss, 'cosine')
        outputs{1} = obj.cosine_fwd(embeddings, W);
      elseif strcmp(obj.loss, 'ACD')
        outputs{1} = obj.ACD_fwd(embeddings, W);
      end
	  
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
	  [sh, sw, sc, sn] = size(inputs{1});
	  embeddings = reshape(gather(inputs{1}), sh*sw*sc, sn);
	  W = gather(inputs{3}); % W is a square matrix which contains graph distances between all the patches in the current batch
	  
      % Case 1: there is exactly one label per data sample n out of sn samples
      embeddings = reshape(gather(inputs{1}), sh*sw*sc, sn);
      
      if strcmp(obj.loss, 'hadsell')
        der = obj.dirLIM_bwd(embeddings, W);
      elseif strcmp(obj.loss, 'hadsell_norm')
        der = obj.dirLIMNormalized_bwd(embeddings, W);
      elseif strcmp(obj.loss, 'cosine')
        der = obj.cosine_bwd(embeddings, W);
      elseif strcmp(obj.loss, 'ACD')
        der = obj.ACD_bwd(embeddings, W);
      end
	
      % Case 1
      derInputs{1} = reshape(der, sh, sw, sc, sn);
      
      derInputs{2} = [] ;
      derInputs{3} = 1 ;
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

    function obj = SemiSupervisedEmbeddingLoss(varargin)
      obj.load(varargin) ;
    end
    
	% Hadsell et al. 2006
	function out = dirLIM_fwd(obj, embeddings, W)
	  obj.D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:i
		  if W(i,j)
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
	
	function derOut = dirLIM_bwd(obj, embeddings, W)
	  Der = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:i
          tmp = 2*(embeddings(:,i) - embeddings(:,j));
          if W(i,j)
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
      % Der = Der; % Maybe Normalize based on the sum of euclideans produced in the forward step? 
      derOut = obj.lambda * Der;
    end
    
    function out = dirLIMNormalized_fwd(obj, embeddings, W)
      % Normalize embeddings first
	  obj.N = ( sqrt(sum(embeddings.^2,1)) );
	  embeddings = embeddings ./ repmat(obj.N, size(embeddings,1), 1);
      
	  obj.D = zeros(size(embeddings,2));
	  for i=1:size(embeddings,2)
		for j=1:i
		  if W(i,j)
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
	
	function derOut = dirLIMNormalized_bwd(obj, embeddings, W)
      % Normalize embeddings first
      embeddings_orig = embeddings;
	  embeddings = embeddings ./ repmat(obj.N, size(embeddings,1), 1);
      
	  Der = zeros(size(embeddings));
	  for i=1:size(embeddings,2)
		for j=1:i
          tmp = 2*(embeddings(:,i) - embeddings(:,j)) .* (obj.N(i) - embeddings(:,i).*embeddings_orig(:,i)) ./ (obj.N(i)^2);
          if W(i,j)
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
    
    function out = cosine_fwd(obj, embeddings, W)
	  obj.D = zeros(size(embeddings,2));
      fac = 1/size(embeddings,2);
	  for i=1:size(embeddings,2)
		for j=1:i
          absum = sum( embeddings(:,i) .* embeddings(:,j) );
          a2sumb2sum = sum( embeddings(:,i).^2 ) * sum( embeddings(:,j).^2 );
          distance = (1 - absum / sqrt(a2sumb2sum));
		  if W(i,j)
			obj.D(i,j) = distance;
          else
			obj.D(i,j) = max(0, obj.margin - distance);
          end
          obj.D(j,i) = obj.D(i,j);
        end
      end
      
      out = fac * sum(obj.D(:));
    end
    
    function derOut = cosine_bwd(obj, embeddings, W)      
      Der = zeros(size(embeddings));
      fac = 1/size(embeddings,2);
	  for i=1:size(embeddings,2)
		for j=1:i
          a = embeddings(:,i);
          b = embeddings(:,j);
          a2sum = sum( a.^2 );
          b2sum = sum( b.^2 );
          sqrta2sumb2sum = sqrt(a2sum*b2sum);
		  if W(i,j)
			Der(:,i) = Der(:,i) - (b./sqrta2sumb2sum - obj.D(i,j) * a./a2sum);
            Der(:,j) = Der(:,j) - (a./sqrta2sumb2sum - obj.D(i,j) * b./b2sum);
          else
			if obj.margin - obj.D(i,j) >= 0
			  Der(:,i) = Der(:,i) + (b./sqrta2sumb2sum - obj.D(i,j) * a./a2sum);
              Der(:,j) = Der(:,j) + (a./sqrta2sumb2sum - obj.D(i,j) * b./b2sum);
			end
          end
		end
      end
	  
	  derOut = fac * obj.lambda .* Der;
    end
    
    
    function out = ACD_fwd(obj, embeddings, W)
	  obj.D = zeros(size(embeddings,2));
      fac = 1/size(embeddings,2);
	  for i=1:size(embeddings,2)
		for j=1:i
          if i==j
              continue;
          end
          absum = sum( embeddings(:,i) .* embeddings(:,j) );
          a2sumb2sum = sum( embeddings(:,i).^2 ) * sum( embeddings(:,j).^2 );
          obj.cossim(i,j) = absum / (sqrt(a2sumb2sum) + obj.epsilon); 
          distance = acos(obj.cossim(i,j)) / pi;
		  if W(i,j)
			obj.D(i,j) = distance;
          else
			obj.D(i,j) = max(0, obj.margin - distance);
          end
          obj.D(j,i) = obj.D(i,j);
        end
      end
      %obj.D = acos(obj.D) ./ pi;
      
      out = fac * sum(obj.D(:));
    end
    
    function derOut = ACD_bwd(obj, embeddings, W)
      Der = zeros(size(embeddings));
      fac = 1/size(embeddings,2);
	  for i=1:size(embeddings,2)
		for j=1:i
          if i == j
              continue;
          end
          a = embeddings(:,i);
          b = embeddings(:,j);
          a2sum = sum( a.^2 ) + obj.epsilon;
          b2sum = sum( b.^2 ) + obj.epsilon;
          sqrta2sumb2sum = sqrt(a2sum*b2sum);
		  if W(i,j)
            if obj.cossim(i,j) == 1 % Clipping the extreme cases...
              continue;
            end
			Der(:,i) = Der(:,i) + (-1/(sqrt(1-(obj.cossim(i,j)^2)) + obj.epsilon))*(b./sqrta2sumb2sum - obj.cossim(i,j) * a./a2sum);
            Der(:,j) = Der(:,j) + (-1/(sqrt(1-(obj.cossim(i,j)^2)) + obj.epsilon))*(a./sqrta2sumb2sum - obj.cossim(i,j) * b./b2sum);
          else
			if obj.margin - obj.D(i,j) >= 0
              if obj.cossim(i,j) == 1 % Clipping the extreme cases...
                continue;
              end
			  Der(:,i) = Der(:,i) - (-1/(sqrt(1-(obj.cossim(i,j)^2)) + obj.epsilon))*(b./sqrta2sumb2sum - obj.cossim(i,j) * a./a2sum);
              Der(:,j) = Der(:,j) - (-1/(sqrt(1-(obj.cossim(i,j)^2)) + obj.epsilon))*(a./sqrta2sumb2sum - obj.cossim(i,j) * b./b2sum);
			end
          end
		end
      end
	  
	  derOut = fac * obj.lambda .* Der;
    end
	
  end
end
