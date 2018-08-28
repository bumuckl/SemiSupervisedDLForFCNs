function Eval = evaluateCC(imagestruct, thresh, radius, scale, savepath, plotROC, plotData)
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
        savepath = '';
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
        Image.labels = round(scale * Image.labels);
        Image.labelmap = zeros(size(Image.data,1), size(Image.data,2));
        for p=1:size(Image.labels,1) 
            Image.labelmap = circleAroundPoint( Image.labelmap, [Image.labels(p,1) Image.labels(p,2)], radius * scale );
        end

        Image.responseMapBinary = Image.responseMap > thresh;
        % Get TP and FP
        Image.responseMapTP = Image.responseMapBinary .* Image.labelmap;
        Image.responseMapFP = Image.responseMapBinary .* (~Image.labelmap);
        Image.responsesTP = labelsFromBinaryMap(Image.responseMapTP);
        Image.responsesFP = labelsFromBinaryMap(Image.responseMapFP);
        Image.responsesAll = [Image.responsesTP; Image.responsesFP];
        % Count connected components in these respective maps
        CCP = bwconncomp(Image.labelmap);
        CCTP = bwconncomp(Image.responseMapTP);
        CCFP = bwconncomp(Image.responseMapFP);
        CCFN = bwconncomp(Image.responseMapTP ~= Image.labelmap);
        CC = bwconncomp(Image.responseMapBinary);
        
        % Draw circles around maxima to be able to see them
        %[rows, cols] = find(Image.responseMapBinary == 1);
        S = regionprops(CC,'Centroid');
        for r=1:length(S) % For too many rows, circleAroundPoint will take forever. So we simply limit the number of circles. 100 FP is enough, anyway
            Image.responseMapBinary = circleAroundPoint( Image.responseMapBinary, S(r).Centroid, radius * scale );
        end
        
        % Visualize
        tmp = Image.data;
        tmp(:,:,2) = Image.labelmap;
        % this is a killing mistake (Debugged by Shadi). 
        tmp(:,:,1) = Image.responseMapBinary';
        
        if saveImage
            imwrite(tmp, [savepath num2str(i) '.png']);
        end
        if plotData
            figure(handle), cla;
            subplot(1,2,1), imshow(tmp);
            subplot(1,2,2), imshow(Image.responseMap);
        end

        % Update overall Eval struct
        Eval.TP = Eval.TP + CCTP.NumObjects;
        Eval.FP = Eval.FP + CCFP.NumObjects;
        Eval.FN = Eval.FN + CCFN.NumObjects;
        Eval.TN = Eval.TN + (size(Image.data,1) * size(Image.data,2)) ...
              - CCTP.NumObjects ... 
              - CCFN.NumObjects ... 
              - CCFP.NumObjects;
        Eval.P = Eval.P + CCP.NumObjects;
        Eval.N = Eval.N + (size(Image.data,1) * size(Image.data,2)) - CCP.NumObjects;
        Eval.Score = [Eval.Score; Image.responseMap(:)];
        Eval.Target = [Eval.Target; Image.labelmap(:)];
        
        % Update patients Eval struct
        Eval.Patient(patient).TP = Eval.Patient(patient).TP + CCTP.NumObjects;
        Eval.Patient(patient).FP = Eval.Patient(patient).FP + CCFP.NumObjects;
        Eval.Patient(patient).FN = Eval.Patient(patient).FN + CCFN.NumObjects;
        Eval.Patient(patient).TN = Eval.Patient(patient).TN + (size(Image.data,1) * size(Image.data,2)) ...
              - CCTP.NumObjects ... 
              - CCFN.NumObjects ... 
              - CCFP.NumObjects;
        Eval.Patient(patient).P = Eval.Patient(patient).P + CCP.NumObjects;
        Eval.Patient(patient).N = Eval.Patient(patient).N + (size(Image.data,1) * size(Image.data,2)) - CCP.NumObjects;
        Eval.Patient(patient).Score = [Eval.Patient(patient).Score; Image.responseMap(:)];
        Eval.Patient(patient).Target = [Eval.Patient(patient).Target; Image.labelmap(:)];
    end

    % Calculate F-score and stuff
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
    end
    
    %Eval.Score = [];
    %Eval.Target = [];
end

