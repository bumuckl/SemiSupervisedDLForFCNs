% DICELOSS
%
% As seen in "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
%
% Author: Christoph Baur
classdef DiceLoss < dagnn.ElementWise
  properties
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    batch = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      p = gather(inputs{1}(:,:,2,:)) > 0.5;
	  g = gather(inputs{2});
      [sh sw sc sn] = size(inputs{1});
      
      if size(p,4) ~= size(g,4)
          error('DiceLoss: Forward: Predictions and Labels must have equal size!');
      end
      
      % Make labels binary
      g = g-1;
	  
      % Computations
      psum = sum(p(:)) / numel(p);
      gsum = sum(g(:)) / numel(g);
      pgsum = sum(p(:).*g(:)) / numel(p);
	  outputs{1} = 1 - (2 * pgsum) / (psum + gsum);  
	  
      %n = obj.numAveraged ;
      %m = n + size(inputs{1},4) ;
      obj.average = (obj.batch * obj.average + gather(outputs{1})) / (obj.batch+1) ;
      obj.batch = obj.batch + 1 ;
      
      figure(5), cla;
      for i=1:size(inputs{1},4)
          img = exp(squeeze(inputs{1}(:,:,:,i)));
          img = img(:,:,2) ./ sum(img,3);
          subplot(size(inputs{1},4),2,2*i-1), imagesc(img);
          subplot(size(inputs{1},4),2,2*i), imagesc(squeeze(inputs{2}(:,:,:,i)));
      end
      drawnow;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
	  p = gather(inputs{1}(:,:,2,:)) > 0.5;
      g = gather(inputs{2});
      [sh sw sc sn] = size(inputs{1});
      
      if size(p,4) ~= size(g,4)
          error('DiceLoss: Forward: Predictions and Labels must have equal size!');
      end
      
      % Make labels binary
      g = g-1;
	  
      % Computations
      derInputs{1} = zeros(size(inputs{1}));
      derInputs{1}(:,:,1,:) = obj.hlp_der(1-p,1-g);
      derInputs{1}(:,:,2,:) = obj.hlp_der(p,g);
      
      disp(' ');
      disp(['DiceLoss Stats: Min: ' num2str(min(derInputs{1}(:))) ' - Max: ' num2str(max(derInputs{1}(:))) ' - Mean:' num2str(mean(derInputs{1}(:))) ' - Median: ' num2str(median(derInputs{1}(:)))]);
      
      derInputs{2} = [] ;
      derParams = {} ;
      
      %for i=1:size(inputs{1},4)
      %    figure(5), cla;
      %    img = exp(squeeze(derInputs{1}(:,:,:,i)));
      %    img(:,:,3) = zeros(size(img,1), size(img,2));
      %    imagesc(img);
      %end
    end
    
    function der = hlp_der(obj,p,g)
        psum = sum(p(:)) / numel(p);
        gsum = sum(g(:)) /numel(g);
        psumgsumsquare = (gsum + psum)^2;
        pgsum = sum(p(:).*g(:)) / numel(p);
      
        der = -2 * ((g .* (psum + gsum)) - 2*pgsum.*p) ./ psumgsumsquare;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
      obj.batch = 0 ;
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

    function obj = DiceLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
