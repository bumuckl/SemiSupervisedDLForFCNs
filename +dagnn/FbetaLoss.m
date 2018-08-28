% FbetaLoss as proposed by Shadi Albarqouni
%
% Author: Christoph Baur
classdef FbetaLoss < dagnn.ElementWise
  properties
    opts = {};
    beta = [1 1];
    updateBeta = true;
    beta_lr = 1;
    beta_weightDecay = 0.0005;
    batchesAverageSize = 20;
    debug = true;
    epsilon = 0.00000000001;
		gradientClipping = false;
		clipAt = 1000.0;
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    batch = 0
    batchesAverage = []
    h_debug = 0
    h_debug2 = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
        global TRAINING;
      p = gather(inputs{1}(:,:,2,:)) > 0.5;
	  g = gather(inputs{2});
      [sh, sw, sc, sn] = size(inputs{1});
      
      if size(p,4) ~= size(g,4)
          error('FbetaLoss: Forward: Predictions and Labels must have equal size!');
      end
      
      % Trick for handling labels that are 0 and hence should be ignored
      g(isnan(g)) = 0;
      validPixels = find(g > 0);
      
      % Make labels binary
      g = g-1;
	  
      % Computations
      n = numel(validPixels) + obj.epsilon;
      psum = sum(p(validPixels)) / n;
      gsum = sum(g(validPixels)) / n;
      pgsum = sum(p(validPixels).*g(validPixels)) / n;
	  outputs{1} = 1 - (2 * pgsum) / (psum + gsum + obj.epsilon); 
      %outputs{1} = 1 - ((1+obj.beta^2) * pgsum) / (psum + (obj.beta^2)*gsum);      
	  
      %n = obj.numAveraged ;
      %m = n + size(inputs{1},4) ;
      obj.average = (obj.batch * obj.average + gather(outputs{1})) / (obj.batch+1) ;
      obj.batch = obj.batch + 1 ;
      if numel(obj.batchesAverage) < obj.batchesAverageSize
          obj.batchesAverage(end+1) = outputs{1};
      else
          obj.batchesAverage(1:end-1) = obj.batchesAverage(2:end);
          obj.batchesAverage(end) = outputs{1};
      end
      figure(10), cla, plot(1:numel(obj.batchesAverage), obj.batchesAverage);
      
      if obj.h_debug == 0
          obj.h_debug = figure;
      else
          figure(obj.h_debug), cla;
      end
      title('FbetaLoss Forward');
      for i=1:size(inputs{1},4)
          img = exp(squeeze(inputs{1}(:,:,:,i)));
          img = img(:,:,2) ./ sum(img,3);
          %img = condense3DMulticlassResponse( inputs{1}(:,:,:,i) );
          subplot(size(inputs{1},4),2,2*i-1), imagesc(img), xlabel('Prediction');
          subplot(size(inputs{1},4),2,2*i), imagesc(squeeze(inputs{2}(:,:,:,i))), xlabel('Groundtruth');
      end
      drawnow;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
	  p = gather(inputs{1}(:,:,2,:)) > 0.5;
	  g = gather(inputs{2});
      if numel(params) > 0 && obj.updateBeta == true
        obj.beta = gather(params{1});
      end
      [sh sw sc sn] = size(inputs{1});
      
      if size(p,4) ~= size(g,4)
          error('FbetaLoss: Forward: Predictions and Labels must have equal size!');
      end
      
      % Trick for handling labels that are 0 and hence should be ignored
      g(isnan(g)) = 0;
      validDers = repmat( g > 0, 1, 1, 2, 1);
      validPixels = find(g > 0);
      
      % Make labels binary
      g = g-1;
	  
      % Computations
      derInputs{1} = zeros(size(inputs{1}));
      [derInputs{1}(:,:,1,:), derParams{1}(1)] = obj.hlp_der(1-p, 1-g, obj.beta(1), validPixels);
      [derInputs{1}(:,:,2,:), derParams{1}(2)] = obj.hlp_der(p, g, obj.beta(2), validPixels);
      derInputs{1} = derInputs{1} .* validDers;
      
      %disp(' ');
      %disp(['FbetaLoss Stats: Min: ' num2str(min(derInputs{1}(:))) ' - Max: ' num2str(max(derInputs{1}(:))) ' - Mean:' num2str(mean(derInputs{1}(:))) ' - Median: ' num2str(median(derInputs{1}(:)))]);
      
      derInputs{2} = [] ;
      if ~obj.updateBeta
        derParams = {0 0};
      end
      
      figure(6), cla;
      for i=1:size(inputs{1},4)
          subplot(size(inputs{1},4),3,3*i-2), imagesc(derInputs{1}(:,:,1,i)), xlabel('derInputs 1');
          subplot(size(inputs{1},4),3,3*i-1), imagesc(derInputs{1}(:,:,2,i)), xlabel('derInputs 2');
          subplot(size(inputs{1},4),3,3*i), imagesc(squeeze(inputs{2}(:,:,:,i))), xlabel('Prediction');
      end
      drawnow;
    end
    
    function [der, derBeta] = hlp_der(obj, p, g, beta, validPixels)
        n = numel(validPixels) + obj.epsilon;
        psum = sum(p(validPixels)) / n;
        gsum = sum(g(validPixels)) / n;
        psumgsumsquare = ((beta^2)*gsum + psum)^2 + obj.epsilon;
        pgsum = sum(p(validPixels).*g(validPixels)) / n;
      
        der = -(1+beta^2) .* ((g .* (psum+(beta^2)*gsum)) - pgsum) ./ psumgsumsquare;
        %disp(['derFbeta Stats: Min: ' num2str(min(der(:))) ' - Max: ' num2str(max(der(:))) ' - Mean:' num2str(mean(der(:))) ' - Median: ' num2str(median(der(:)))]);
        
        derBeta = 0;
        if obj.updateBeta && numel(obj.batchesAverage) == obj.batchesAverageSize && mean(obj.batchesAverage(end-4:end)) <= mean(obj.batchesAverage(1:end-5)) % Only update if the last average is smaller than the last previous averages
            derBeta = (-2*beta * pgsum * (psum - gsum) ./ psumgsumsquare);
            disp(['derBeta: ' num2str(derBeta)]);
        end
				
				% As seen here: http://www.wildml.com/deep-learning-glossary/#gradient-clipping
				if obj.gradientClipping
					l2norm = (der).^2;
					l2norm = sqrt(sum(l2norm(:)));
					der = der .* (obj.clipAt/l2norm);
				end
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
      obj.batch = 0 ;
    end
    
    function params = initParams(obj)
      params{1} = single(obj.beta);
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

    function obj = FbetaLoss(varargin)
      obj.load(varargin) ;
    end
    
    function attach(obj, net, index)
      attach@dagnn.ElementWise(obj, net, index) ;
      if obj.updateBeta
          p = net.getParamIndex(net.layers(index).params{1}) ;
          net.params(p).trainMethod = 'gradientWObatchsize' ;
          net.params(p).learningRate = obj.beta_lr;
          net.params(p).weightDecay = obj.beta_weightDecay;
      end
    end
  end
end
