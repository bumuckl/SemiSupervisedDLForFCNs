function Eval = evaluateSingleMax(single_imagestruct, radius, scale, savepath, plotData)
%EVALUATESINGLEMAX Calculate f-score etc for a structure
%containing a single image, its responsemap and its labelmap
%
%
% @Author: Christoph Baur

    if nargin < 4
        savepath = '';
    end
    if nargin < 5
        plotData = 0;
    end

    % Set format to long
    format long;
    
    % Find best threshold for a struct containing only one image
    imagestruct = {};
    imagestruct{1} = single_imagestruct;
    
    % Do the actual evaluation and save results
    if nargin > 3
        Eval = CNN.evaluateMax(imagestruct, radius, scale, savepath, plotData);
    else
        Eval = CNN.evaluateMax(imagestruct, radius, scale);
    end
end

