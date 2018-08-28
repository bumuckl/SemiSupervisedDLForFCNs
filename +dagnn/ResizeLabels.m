% Take both the input data and the corresponding labels, which are also
% supposed to be images. Rescale the label images such that they have the
% same size as their data counterparts. Required for a FCN to work.
%
% (c) 2016 Christoph Baur

classdef ResizeLabels < dagnn.Layer
  methods
    function outputs = forward(obj, inputs, params)
        [sh sw sc sn] = size(inputs{1});
        labels = gather(inputs{2});
        
        tmp = zeros(sh, sw, 1, sn);
        for i=1:sn
           img = squeeze(labels(:,:,1,i));
           tmp(:,:,1,i) = imresize(img, [sh sw], 'nearest'); 
        end
        
%         figure(1), cla;
%         for i=1:sn
%             subplot(2,sn,i), imagesc(labels(:,:,:,i));
%             subplot(2,sn,i+sn), imagesc(tmp(:,:,:,i));
%         end
        
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
