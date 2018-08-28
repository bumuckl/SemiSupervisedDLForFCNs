% Based on labels or label images, compute an instanceMatrix with class
% weights for every label which can be used by dagnn.Loss
%
% (c) 2016 Christoph Baur

classdef ResizeLabels < dagnn.Layer
  methods
    function outputs = forward(obj, inputs, params)
        [sh sw sc sn] = size(inputs{1});
        labels = gather(inputs{2});
        
        tmp = zeros(sh, sw, 1, sn);
        for i=1:size(labels,3)
           tmp(:,:,1,i) = imresize(squeeze(labels(:,:,1,i)), [sh sw], 'nearest'); 
        end
        
        outputs = {tmp};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
        % Just feed the derivatives of the next layer backwards, this layer does not change them
        derInputs = {};
        derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      xSize = inputSizes{1};
      gSize = inputSizes{2};
      outputSizes{1} = {1};
    end

    function obj = ResizeLabels(varargin)
      obj.load(varargin);
    end
  end
end
