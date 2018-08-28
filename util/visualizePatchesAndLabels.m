function [dl_patches, dl_labels] = visualizePatchesAndLabels(patches, labels)
%VISUALIZEPATCHESANDLABELS Visualize patches and labels for verification
%purposes
%
% @Author: Christoph Baur

    h = figure;
    for f=1:length(patches)
        for p=1:length(patches{f})            
            figure(h), cla;
            imshow(patches{f}{p});
            xlabel(['Label: ' num2str(labels{f}{p})]);
            
            pause;
        end
    end
end

