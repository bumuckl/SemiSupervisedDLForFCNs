function thresh = getOptimalThreshold( P, G, decimals )
%GETOPTIMALTHRESHOLD Given a tensor of predictions, corresponding
%ground-truth and a umber of decimals for precision of the threshold, find
%the best threshold on P
%   
% Author: Christoph Baur

    thresh = recursive_getOptimalThreshold(double(P), double(G), 0, 1, 0.1, decimals);
end

function thresh = recursive_getOptimalThreshold(P, G, lowerBound, upperBound, stepsize, remainingIterations)
    scores = [];
    threshs = lowerBound:stepsize:upperBound;
    for i=1:numel(threshs)
        scores(i) = Dice(P > threshs(i), G);
    end
    [~, threshIdx] = max(scores);
    lowerBound = threshs(max(1, threshIdx-1));
    upperBound = threshs(min(numel(threshs), threshIdx+1));
    thresh = threshs(threshIdx);
    
    if remainingIterations > 0
        thresh = recursive_getOptimalThreshold(P, G, lowerBound, upperBound, stepsize/10, remainingIterations-1);
    end
end