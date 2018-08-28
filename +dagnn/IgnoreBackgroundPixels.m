% Take both the input data and the corresponding labels, which are also
% supposed to be images. Rescale the label images such that they have the
% same size as their data counterparts. Required for a FCN to work.
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

classdef IgnoreBackgroundPixels < dagnn.Layer
  properties
    threshold = 0.1
  end
    
  methods
    function outputs = forward(obj, inputs, params)
        [sh sw sc sn] = size(inputs{1});
        data = gather(inputs{1});
        labels = gather(inputs{2});
        
        data(data < obj.threshold) = nan; % for domain A it was 0.01
        data = sum(data, 3);
        labels(isnan(data)) = 0;
        
        outputs = {labels};
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

    function obj = IgnoreBackgroundPixels(varargin)
      obj.load(varargin);
    end
  end
end
