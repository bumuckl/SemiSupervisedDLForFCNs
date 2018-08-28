function [Eval, thresh] = evaluateSingle(single_imagestruct, radius, scale, savepath, plotROC, plotData)
%EVALUATESINGLE Calculate f-score etc for a structure
%containing a single image, its responsemap and its labelmap
%
%
% @Author: Christoph Baur

    if nargin < 4
        savepath = '';
    end
    if nargin < 5
        plotROC = 0;
    end
    if nargin < 6
        plotData = 0;
    end

    % Set format to long
    format long;
    
    % Init vars
    thresh_start = 0.50;
    thresh_step = 0.001;
    thresh_end = 1.0;
    
    % Find best threshold for a struct containing only one image
    imagestruct = {};
    imagestruct{1} = single_imagestruct;
    [thresh, Eval] = CNN.findthreshold(imagestruct, thresh_start, thresh_step, thresh_end, radius, scale);
    
    % Do the actual evaluation and save results
    if nargin > 3
        Eval = CNN.evaluate(imagestruct, thresh, radius, scale, savepath, plotROC, plotData);
    else
        Eval = CNN.evaluate(imagestruct, thresh, radius, scale);
    end
end

