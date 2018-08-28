function handle = visualize( imdb, autopause )
%VISUALIZE 

    if nargin < 2
        autopause = 0;
    end
    
    % Variables
    [sh, sw, sc, sn] = size(imdb.images.data);
    [lh, lw, lc, ln] = size(imdb.images.labels);
    disp(['Data size: ' num2str(sh) 'x' num2str(sw) 'x' num2str(sc) 'x' num2str(sn)]);
    disp(['Labels size: ' num2str(lh) 'x' num2str(lw) 'x' num2str(lc) 'x' num2str(ln)]);

    handle = figure, colormap(gray);
    %imdb.images.data = double(normalize2(imdb.images.data));
    for i=1:size(imdb.images.data, 4)
        figure(handle), cla;
        
        if isfield(imdb.images, 'filenames') && length(imdb.images.filenames) > 0
             img = imread([imdb.meta.pathstr '/' imdb.meta.name '/' imdb.images.filenames{i}]);
             imshow(img);
        else
            img = double(squeeze(imdb.images.data(:,:,:,i)));
            numChannels = size(img,3);
            if sh == lh && sw == lw
                label_img = normalize2(squeeze(imdb.images.labels(:,:,1,i)));
            end
            for c=1:numChannels
                channel = squeeze(img(:,:,c));
                subplot(numChannels,3,c), imshow(channel), xlabel(['Channel ' num2str(c)]), hold on;
            end
            img_overlay = zeros(sh, sw, 3);
            img_overlay(:,:,1) = img(:,:,1);
            img_overlay(:,:,3) = label_img;
            subplot(numChannels,3,[numChannels*1+1,numChannels*1+numChannels]), imshow(label_img), xlabel('Ground-truth');
            subplot(numChannels,3,[2*numChannels+1,2*numChannels+numChannels]), imshow(img_overlay), xlabel('Overlay');
            
            labelmap_raw = squeeze(imdb.images.labels(:,:,1,i));
            pospixels = sum(labelmap_raw(:) == 2) / numel(labelmap_raw);
            disp(['Percentage of positive pixels: ' num2str(pospixels)]);
        end
        
        % Meta information
        disp(['Min: ' num2str(min(img(:))) ' - Max:' num2str(max(img(:)))]);
        
        gt = 'n/a';
        try
            gt = num2str(imdb.images.labels_gold(i));
        catch ME
            %disp('No gold label available');
        end
        
        label = 'n/a';
        try
            label = num2str(imdb.images.labels(i));
        catch ME
            disp('No label available');
        end
        
        set = 'n/a';
        try
            set = num2str(imdb.images.set(i));
        catch
            disp('No set available');
        end
        
        try
            if (sh == 1 && sw == 1)
                xlabel(['Label: ' label ' - Set: ' set ' - GT: ' gt]);
            end
        catch ME
            disp('Label could not be determined');
        end
        
        if autopause
            pause(1000);
        else
            pause;
        end
    end
    
end

