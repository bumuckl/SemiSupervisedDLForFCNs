function labelmap = binaryMapFromLabels( labels, imagesize, circleradius, numWhitePixelsPerCircle )
%BINARYMAPFROMLABELS Given a matrix of label coordinates, return a binary
%map of labels, possibly with a circle area around

    if nargin < 4
        numWhitePixelsPerCircle = -1;
    end

    labelmap = zeros(imagesize(1), imagesize(2));
    for p=1:size(labels,1)
        if numWhitePixelsPerCircle > 0
            labelmap = circleAroundPoint( labelmap, [labels(p,1) labels(p,2)], circleradius, numWhitePixelsPerCircle );
        else
            labelmap = circleAroundPoint( labelmap, [labels(p,1) labels(p,2)], circleradius );
        end
    end
end

