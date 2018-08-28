classdef SoftmaxLogLoss < dagnn.ElementWise
% SOFTMAXLOGLOSS is just a stub for the MatConvNet default softmaxlog-loss,
% however it filters the input batch. Any patches with NaN-labels are
% removed.
%
% Author: Christoph Baur
  properties
	b = -1;
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      % Filter the input for NaN-labeled patches and remove them (to speed
      % up a bit)
      unlabeled_idx = find(isnan(inputs{2}));
      labeled_idx = setdiff(1:length(inputs{2}),unlabeled_idx);
      inputs{1} = inputs{1}(:,:,:,labeled_idx);
      inputs{2} = inputs{2}(labeled_idx);
        
      outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', 'softmaxlog', obj.opts{:}) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      % Filter the input for NaN-labeled patches and set them to zero
      % Simply because if a label is zero, vl_nnloss will skip it
      inputs{2}(isnan(inputs{2})) = 0;
      
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', 'softmaxlog', obj.opts{:}) ;
      %figure(5), subplot(1,2,2), imagesc(squeeze(derInputs{1})), xlabel('SoftmaxLogLoss Derivatives'), drawnow;
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

    function obj = SoftmaxLogLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
