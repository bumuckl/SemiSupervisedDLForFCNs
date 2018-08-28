function Eval = evaluate(imagestruct, thresh, radius, scale, savepath, plotROC, plotData)
%FINDTHRESHOLD Given a threshold, caluclate f-score etc for all images
%(responsemaps and labelmaps) inside imagestruct
%
% @Author: Christoph Baur

    % Set format to long
    format long;

    % Init Eval struct
    Eval = CNN.EvalStruct;
    Eval.Patient(12) = CNN.EvalStruct;
    
    if nargin < 5
        savepath = '.';
    end
    if nargin < 6
        plotROC = 0;
    end
    if nargin < 7
        plotData = 1;
    end
    if nargin > 4 && length(savepath) > 0
        saveImage = 1;
    else
        saveImage = 0;
    end
    if plotData
        handle = figure;
    end

    % Iterate over all images
    for i=1:length(imagestruct)
        if isempty(imagestruct{i})
           continue; 
        end
        Image = imagestruct{i};
        
        % Hardcoded part: we manually map the indices to patients
        % Indices 2 and 6: Patient 9
        % Indices 21 to 33: Patient 11
        % Indices 34-37, 39-41: Patient 12
        if i < 21
            patient = 9;
        elseif i >= 21 && i <= 33
            patient = 11;
        else
            patient = 12;
        end
        
        
        if isempty(Eval.Patient(patient).Score)
            Eval.Patient(patient) = CNN.EvalStruct;
        end
        
        % Create labelmap according to radius
        if ~isfield(Image, 'labels')
            Image.labels = [];
        end
        Image.labels = scale * Image.labels;
        Image.labelmap = zeros(size(Image.data,1), size(Image.data,2));
        for p=1:size(Image.labels,1)
            Image.labelmap = circleAroundPoint( Image.labelmap, [Image.labels(p,1) Image.labels(p,2)], 1 );
        end

        Image.responseMapBinary = Image.responseMap > thresh;
        % TODO: Non maximum suppression
        
        % Extract all points from responseMapBinary and categorize them
        % into TP, FP
        Image.responsesAll = labelsFromBinaryMap(Image.responseMapBinary);
        Image.responsesTP = [];
        Image.responsesFP = [];
        Image.labelsWithResponses = zeros(1,size(Image.labels,1));
        if length(Image.labels) == 0
            Image.responsesFP = Image.responsesAll;
        else
            for p=1:size(Image.responsesAll,1)
                % For the current response point, calculate the euclidean
                % distance to any labels. If it is smaller than radius*scale
                % for one label, we have a TP. Otherwise we have a FP
                point = Image.responsesAll(p,:);
                pmat = repmat(point, size(Image.labels,1), 1);
                distances = sqrt(sum((pmat - Image.labels).^2, 2));
                [min_distance, corresponding_label] = min(distances);
                if min(distances) <= radius*scale
                    Image.responsesTP = [Image.responsesTP; point];
                else
                    Image.responsesFP = [Image.responsesTP; point];
                end
                Image.labelsWithResponses(corresponding_label(1)) = 1;
            end
        end
        
        % Get TP and FP
        Image.responseMapTP = binaryMapFromLabels( Image.responsesTP, size(Image.responseMapBinary), 1);
        Image.responseMapFP = binaryMapFromLabels( Image.responsesFP, size(Image.responseMapBinary), 1);
        Image.responsesAll = [Image.responsesTP; Image.responsesFP];
        
        % For the FP, compute all the connected components
        CCFP = bwconncomp(Image.responseMapFP);
        
        % Visualize
        tmp = Image.data;
        tmp(:,:,2) = Image.labelmap;
        tmp(:,:,1) = Image.responseMapBinary;
        if saveImage
            imwrite(tmp, [savepath num2str(i) '.png']);
        end
        if plotData
            figure(handle), cla;
            subplot(1,2,1), imshow(tmp);
            subplot(1,2,2), imshow(Image.responseMap);
        end

        % Update Overall Eval struct
        Eval.TP = Eval.TP + sum(Image.labelsWithResponses == 1);
        Eval.FP = Eval.FP + CCFP.NumObjects;
        Eval.FN = Eval.FN + sum(Image.labelsWithResponses == 0);
        Eval.TN = Eval.TN + (size(Image.data,1) * size(Image.data,2)) ...
              - sum(Image.responseMapFP(:)) ... 
              - sum(sum(Image.labelsWithResponses == 0)) ... 
              - sum(Image.responseMapTP(:));
        Eval.P = Eval.P + sum(Image.labelmap(:));
        Eval.N = Eval.N + sum(~Image.labelmap(:));
        Eval.Score = [Eval.Score; Image.responseMap(:)];
        Eval.Target = [Eval.Target; Image.labelmap(:)];
        
        % Update patientwise Eval struct
        Eval.Patient(patient).TP = Eval.Patient(patient).TP + sum(Image.labelsWithResponses == 1);
        Eval.Patient(patient).FP = Eval.Patient(patient).FP + CCFP.NumObjects;
        Eval.Patient(patient).FN = Eval.Patient(patient).FN + sum(Image.labelsWithResponses == 0);
        Eval.Patient(patient).TN = Eval.Patient(patient).TN + (size(Image.data,1) * size(Image.data,2)) ...
              - sum(Image.responseMapFP(:)) ... 
              - sum(Image.labelsWithResponses == 0) ... 
              - sum(Image.responseMapTP(:));
        Eval.Patient(patient).P = Eval.Patient(patient).P + sum(Image.labelmap(:));
        Eval.Patient(patient).N = Eval.Patient(patient).N + sum(~Image.labelmap(:));
        Eval.Patient(patient).Score = [Eval.Patient(patient).Score; Image.responseMap(:)];
        Eval.Patient(patient).Target = [Eval.Patient(patient).Target; Image.labelmap(:)];
    end

    % Calculate overall F-score and stuff
    Eval.precision = Eval.TP/(Eval.TP + Eval.FP); % TP/(TP+FP)
    Eval.recall = Eval.TP/(Eval.TP + Eval.FN); % TP/(TP+FN)
    Eval.fscore = (2 * Eval.precision * Eval.recall)/(Eval.precision + Eval.recall);
    Eval.accuracy = (Eval.TP + Eval.TN) / (Eval.P + Eval.N);
    if isnan(Eval.fscore)
        Eval.fscore = 0;
    end
    if isnan(Eval.accuracy)
        Eval.accuracy = 0;
    end
    
    % Calculate patientwise F-score and stuff
    for patient=1:length(Eval.Patient)
        if isempty(Eval.Patient(patient).Score)
            continue;
        end
        
        Eval.Patient(patient).precision = Eval.Patient(patient).TP/(Eval.Patient(patient).TP + Eval.Patient(patient).FP); % TP/(TP+FP)
        Eval.Patient(patient).recall = Eval.Patient(patient).TP/(Eval.Patient(patient).TP + Eval.Patient(patient).FN); % TP/(TP+FN)
        Eval.Patient(patient).fscore = (2 * Eval.Patient(patient).precision * Eval.Patient(patient).recall)/(Eval.Patient(patient).precision + Eval.Patient(patient).recall);
        Eval.Patient(patient).accuracy = (Eval.Patient(patient).TP + Eval.Patient(patient).TN) / (Eval.Patient(patient).P + Eval.Patient(patient).N);
        if isnan(Eval.Patient(patient).fscore)
            Eval.Patient(patient).fscore = 0;
        end
        if isnan(Eval.Patient(patient).accuracy)
            Eval.Patient(patient).accuracy = 0;
        end
        %Eval.Patient(patient).Score = [];
        %Eval.Patient(patient).Target = [];
    end
    
    if plotROC && ~isempty(Eval.Score) && ~isempty(Eval.Target)
        prec_rec(Eval.Score, Eval.Target);
        %savefig([savepath '/ROC.fig']);
    end
    
    %Eval.Score = [];
    %Eval.Target = [];
end

