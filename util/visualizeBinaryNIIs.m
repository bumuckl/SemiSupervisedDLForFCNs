function [ h ] = visualizeBinaryNIIs( prediction_nii, groundtruth_nii )

    prediction_data = prediction_nii.getData();
    groundtruth_data = groundtruth_nii.getData();
    
    % Create volumes of TP, FP, FN
    tp_vol = prediction_data .* groundtruth_data;
    fp_vol = prediction_data .* (1-groundtruth_data);
    fn_vol = (1-prediction_data) .* groundtruth_data;
    
    labels = nii_labels.getData();
    [Xtp, Ytp, Ztp] = ind2sub(size(tp_vol), find(tp_vol > 0));
    [Xfp, Yfp, Zfp] = ind2sub(size(fp_vol), find(fp_vol > 0));
    [Xfn, Yfn, Zfn] = ind2sub(size(fn_vol), find(fn_vol > 0));
    
    h = figure;
    cla;
    scatter3(Xtp, Ytp, Ztp, 5, 'g'), hold on; % TP
    scatter3(Xfp, Yfp, Zfp, 5, 'y'), hold on; % FP
    scatter3(Xfn, Yfn, Zfn, 5, 'r'), hold on; % FN
    drawnow;

end

