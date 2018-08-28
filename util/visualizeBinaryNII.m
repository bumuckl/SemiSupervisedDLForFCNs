function [ h ] = visualizeBinaryNII( nii )

    data = nii.getData();
    [Xtp, Ytp, Ztp] = ind2sub(size(data), find(data > 0));
    
    h = figure;
    cla;
    scatter3(Xtp, Ytp, Ztp, 5, 'g'), hold on; % TP
    drawnow;

end

