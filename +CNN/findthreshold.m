function [thresh, Eval] = findthreshold(imagestruct, thresh_start, thresh_step, thresh_end, radius, scale)
%FINDTHRESHOLD Find the best threshold to apply to the responseMaps obtained from predictions on the
% testing images
%
% @Author: Christoph Baur

    % Set format to long
    format long;

    % Iterate over all possible thresholds and find the best one
    best_thresh = 0;
    best_fscore = 0;
    for thresh = thresh_start:thresh_step:thresh_end

        Eval = CNN.evaluate(imagestruct, thresh, radius, scale, '', false, false);

        disp(['Score for thresh = ' num2str(thresh) ': ' num2str(Eval.fscore) ' (Current Best: ' num2str(best_fscore) ')']);
        if Eval.fscore > best_fscore
            best_thresh = thresh;
            best_fscore = Eval.fscore;
        end
    end

    thresh = best_thresh;
end

