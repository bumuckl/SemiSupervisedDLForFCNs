classdef SoftMax < dagnn.ElementWise
  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnsoftmax(inputs{1}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1}) ;
      %disp(['Softmax Stats: Min: ' num2str(min(derInputs{1}(:))) ' - Max: ' num2str(max(derInputs{1}(:))) ' - Mean:' num2str(mean(derInputs{1}(:))) ' - Median: ' num2str(median(derInputs{1}(:)))]);
      derParams = {} ;
    end

    function obj = SoftMax(varargin)
      obj.load(varargin) ;
    end
  end
end
