%% NII
% Class which represents a NII (Nifti) volume and provides convenience
% methods for visualization, extracting slices etc. Requires SPM12.
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

classdef NII < handle
    
    properties
        filename = '';
        name = '';
        nii = [];
        vol = [];
    end
    
    methods
        function obj = NII(fname)
            obj.filename = fname;
            obj.nii = struct;
            obj.vol = spm_vol(fname);
            obj.nii.img = spm_read_vols(obj.vol);
            
            % Convert NaN to 0
            obj.nii.img(isnan(obj.nii.img(:))) = 0;
            
            [~, obj.name, ~] = fileparts(obj.filename);
        end
        
        function volume = getData(self)
            volume = self.nii.img;
        end
        
        function setData(self, data)
            self.nii.img = data;
        end
        
        function meta = getMachine(self)
            meta = self.nii.machine;
        end
        
        function square(self)
            self.nii.img = self.nii.img.^2;
        end
        
        function sub(self, otherNII)
            self.nii.img = self.getData() - otherNII.getData();
        end
        
        function img = getSlice(self, i, axis)
            if nargin < 3
                axis = 1;
            end
            
            if axis == 1 % Sagittal
                img = squeeze(self.nii.img(i,:,:));
            elseif axis == 2 % Coronal
                img = squeeze(self.nii.img(:,i,:));
            elseif axis == 3 % Axial
                img = squeeze(self.nii.img(:,:,i));
            else
                error('Invalid Axis');
            end
        end
		
		function setSlice(self, axis, i, data)
			if axis == 1 % Sagittal
                self.nii.img(i,:,:) = data;
            elseif axis == 2 % Coronal
                self.nii.img(:,i,:) = data;
            elseif axis == 3 % Axial
                self.nii.img(:,:,i) = data;
            else
                error('Invalid Axis');
            end
        end
        
        function permuteDimensions(self, order)
           self.nii.img = permute(self.nii.img, order); 
        end
        
        function rot90k(self, k)
           self.nii.img  = rot90(self.nii.img, k);
        end
        
        function stats(self)
           quickstats(self.getData());
        end
		
		function save(self, filename)
			%save_nii(self.nii, filename);
            self.vol.fname = filename;
            self.vol.dim = size(self.nii.img);
            spm_write_vol(self.vol,self.nii.img);
		end
        
        function sz = size(self)
           sz = size(self.nii.img);
        end
        
        function val = min(self)
           val = min(self.nii.img(:)); 
        end
        
        function val = max(self)
           val = max(self.nii.img(:)); 
        end
        
        function val = mean(self)
           val = mean(self.nii.img(:)); 
        end
        
        function exportSlices(self, axis, targetDir)
            if ~exist(targetDir, 'dir')
                mkdir(targetDir);
            end
            
            for i=1:size(self.nii.img, axis)
                imwrite(self.getSlice(i,axis), ['/' self.name '_' num2str(i) '.png']);
            end
        end
        
        function view(self)
           %view_nii(self.nii); 
        end
        
        function threshold(self, t)
            self.nii.img = self.nii.img > t;
        end
        
        function removeSkull(self, mask)
           self.nii.img(~logical(mask)) = 0; 
        end
        
        function normalize(self, precision)
            if nargin < 2
                precision = 'double';
            end
    
            I = self.nii.img;
            if strcmp(precision, 'double')
                I_norm = double(I);
            elseif strcmp(precision, 'single')
                I_norm = single(I);
            else
                I_norm = double(I);
            end
    
            I_norm = I_norm - min(min(I_norm(:)), 0);
            self.nii.img = I_norm ./ max(I_norm(:));
        end
        
        function compareToNII(self, otherNII, axis, pausetime)
            h = figure;
            colormap(gray);
            for i=1:size(self.nii.img, axis)
                figure(h), cla, 
                subplot(1,2,1), imagesc(self.getSlice(i,axis)), xlabel([self.name ' - Slice ' num2str(i)]);
                subplot(1,2,2), imagesc(otherNII.getSlice(i,axis)), xlabel([otherNII.name ' Slice ' num2str(i)]);
                if nargin < 4
                    pause;
                else
                    pause(pausetime)
                end
            end
        end
        
        function animateSlices(self, axis, pausetime, labels)
            uselabels = true;
            if nargin < 3
                uselabels = false;
            end
            if nargin < 2
                pausetime = 0;
                axis = 1;
            end
            
            h = figure;
            colormap(gray);
            for i=1:size(self.nii.img, axis)
                tmp = self.getSlice(i,axis);
                if uselabels
                   tmp = repmat(tmp, [1, 1, 3]);
                   tmp(:,:,1) = double(labels.getSlice(i,axis));
                end
                figure(h), cla, imagesc(tmp), xlabel(['Slice ' num2str(i)]);
                if nargin < 3 || pausetime == 0
                    pause;
                else
                    pause(pausetime)
                end
            end
        end
        
    end
    
end

