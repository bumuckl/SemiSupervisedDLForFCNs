function [ Images ] = computeResponsemaps_woCSV( net, files, opts )
%COMPUTERESPONSEMAPS For all images in the cell array files, compute their
%responsemaps

    if nargin < 3
       opts = struct;
       opts.rotations = [0, 90, 180, 270];
       opts.mirroring = 0;
       opts.d = 10; % radius around a labeled point for pixels also being considered to have this label (in the original, fullscale image!)
       opts.imageScale = 0.33; % How to scale the input images
       opts.patchsize = [33, 33]; % How big the patches to extract around each label (important for the network)
       opts.responsemapsPath = './results_coarse/';
    end

    numTestImages = -1;
    Images = {};
    if numTestImages < 1
        numTestImages = length(files);
    end
    for f=1:min(length(files),numTestImages)    
        % Read it
        Image = struct;
        Image.data = imread(files{f});

%        
%         % Read corresponding CSV file and obtain labels, create a labelmap
%         [pathstr,name,ext] = fileparts(files{f});
%         if ~exist([pathstr '/' name '.csv'], 'file')
%             continue;
%         end
%         Image.labels = csvread([pathstr '/' name '.csv']);
%         Image.labelmap = zeros(size(Image.data,1), size(Image.data,2));
%         for p=1:size(Image.labels,1)
%            Image.labelmap = circleAroundPoint( Image.labelmap, [Image.labels(p,1) Image.labels(p,2)], opts.d );
%         end

        % Preprocess image
        Image.data = ColorNorm.normalizeStaining(Image.data);
        Image.data = normalize2(Image.data, 'single', 8);
        Image.data = imresize(Image.data, opts.imageScale);
        Image.data_padded = padarray(Image.data, floor(opts.patchsize ./ 2), 'symmetric');
        %Image.labelmap = imresize(Image.labelmap, opts.imageScale);

        % Create directory for saving temporary responsemaps
        mkdir([opts.responsemapsPath num2str(f)]);

        tmp_responsemaps = zeros(size(Image.data,1), size(Image.data,2), 0);
        for r=1:length(opts.rotations)
            % Continue if the output file already exists (This blocks ensures
            % that youc an easily stop and resume creating responseMaps)
            num = (1+opts.mirroring)*r - opts.mirroring;
            if exist([opts.responsemapsPath num2str(f) '/' num2str(num) '.mat'], 'file') == 2
                load([opts.responsemapsPath num2str(f) '/' num2str(num) '.mat']);
                tmp_responsemaps(:,:,end+1) = responsemap;
                if opts.mirroring
                    load([opts.responsemapsPath num2str(f) '/' num2str(num+1 ) '.mat']);
                    tmp_responsemaps(:,:,end+1) = responsemap;
                end
                continue;
            end

            tmp_img = imrotate(Image.data, opts.rotations(r), 'crop');
            tmp_responsemap = CNN.classifyImage2(net, tmp_img, 33 );
            tmp_responsemaps(:,:,end+1) = imrotate(tmp_responsemap, -opts.rotations(r), 'crop');
            tmp_responsemaps(:,:,end) = imfilter(tmp_responsemaps(:,:,end), fspecial('disk', opts.imageScale * opts.d));

            % Save the responsemap to disk
            responsemap = squeeze(tmp_responsemaps(:,:,end));
            imwrite(responsemap, [opts.responsemapsPath num2str(f) '/' num2str(num) '.png']); 
            save([opts.responsemapsPath num2str(f) '/' num2str(num) '.mat'], 'responsemap');

            if opts.mirroring
                tmp_img = flipdim(tmp_img, 2);
                tmp_responsemap = CNN.classifyImage2(net, tmp_img, 33);
                tmp_responsemaps(:,:,end+1) = imrotate(flipdim(tmp_responsemap,2), -opts.rotations(r), 'crop');
                tmp_responsemaps(:,:,end) = imfilter(tmp_responsemaps(:,:,end), fspecial('disk', opts.imageScale * opts.d));

                % Save the responsemap to disk
                responsemap = squeeze(tmp_responsemaps(:,:,end));
                imwrite(responsemap, [opts.responsemapsPath num2str(f) '/' num2str(num+1) '.png']); 
                save([opts.responsemapsPath num2str(f) '/' num2str(num+1) '.mat'], 'responsemap');
            end

        end

        % Save all the responsemaps to disk for inspection

        for i=1:size(tmp_responsemaps,3)
           responsemap = squeeze(tmp_responsemaps(:,:,i));
           imwrite(responsemap, [opts.responsemapsPath num2str(f) '/' num2str(i) '.png']); 
           save([opts.responsemapsPath num2str(f) '/' num2str(i) '.mat'], 'responsemap');
        end

        % Classify with coarse CNN
        Image.responseMap = mean(tmp_responsemaps, 3);
        % Opening with disk element with radius d, followed by nonmaximum suppression
        %Image.responseMap = imfilter(Image.responseMap, fspecial('disk', opts.imageScale * opts.d));

        % Update Images cell array and labels
        Images{f} = Image;
    end

end

