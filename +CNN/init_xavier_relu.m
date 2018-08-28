% INIT_XAVIER_RELU Initialize the weights with help of xaviers improved method
% for relus
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

function filters = init_xavier_relu(h,w,in,out,type)
    if nargin < 5
        type = 'single';
    end
    
    variance = sqrt(2/(h*w*out));
    filters = randn(h,w,in,out,type)*variance;
end