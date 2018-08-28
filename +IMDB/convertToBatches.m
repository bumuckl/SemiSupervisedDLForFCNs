function imdb = convertToBatches( querypath, queryext, targetpath, batchsize, partitions )
%REPARTITION Read all imdb files inside the querypath and split them into
%single imdb structs of batchsize patches each. Store all of them inside
%targetpath
%
% INPUT
%
%   querypath = the path that contains all the desired imdb.mat files, i.e.
%   ../../draft/Mitosis Data/training
%   queryext = the file extnesions of files to look for, i.e. '*.imdb.mat'
%   targetpath = the folder where to store the resulting batch-sized
%   imdb.mat files, i.e. './data/mitosis_101x101_r10_s1'
%   partitions = partitioning of the data into training, validation and
%   testing set.
%
% @Author: Christoph Baur
    
   % Get all the files
   files = getAllFiles(querypath, queryext, 1);
   mkdir([targetpath '.imdb']);
   
   % Create empty imdb and GoGoGo
   imdb = IMDB.init();
   batchpaths = {};
   f = 1;
   batch = 1;
   p = 1;
   while true % This is kind of a state machine
       if ~exist('cimdb') || p > length(cimdb.images.labels)
           if f > length(files)
               break;
           end
           disp(['Loading file ' files{f}]);
           cimdb = IMDB.load(files{f});
           p = 1;
           f = f + 1;
       end
       
       if length(imdb.images.labels) == batchsize
           % Postprocessing
           imdb.meta.classes = unique(imdb.images.labels);
           imdb = IMDB.repartition(imdb, partitions);
           imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
           % Save it
           batchpaths{end+1} = [targetpath '.imdb/' num2str(batch) '.imdb.mat'];
           save(batchpaths{end}, 'imdb');
           disp(['Done with batch ' num2str(batch)]);
           % Reset variables
           batch = batch+1;
           imdb = IMDB.init();
       end
       
       imdb.images.data(:,:,:,end+1) = cimdb.images.data(:,:,:,p);
       imdb.images.labels(1,end+1) = cimdb.images.labels(1,p);
       p = p + 1;
       
       if p + batchsize - 1 > length(imdb.images.labels) && f > length(files)
           break;
       end
       
   end
   
   % Create the imdb which holds all the paths to the actual imdb batches
   % inside the data field
   imdb = IMDB.init();
   imdb.images.subimdbs = batchpaths;
   save([targetpath '.imdb.mat'], 'imdb');
end

