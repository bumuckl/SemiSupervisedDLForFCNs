% Creates the UNET network architecture for medical image segmentation
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

% Options
numClasses = 2;
numInputChannels = 3;

net = dagnn.DagNN();

% 1

block = dagnn.Conv('size', [3 3 numInputChannels 64], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d0a-b', block, {'input'}, {'conv_d0a-b'}, {'conv_d0a-b_filter', 'conv_d0a-b_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d0a-b_relu', block, {'conv_d0a-b'}, {'conv_d0a-b_relu'}, {});

block = dagnn.Conv('size', [3 3 64 64], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d0b-c', block, {'conv_d0a-b_relu'}, {'conv_d0b-c'}, {'conv_d0b-c_filter', 'conv_d0b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d0b-c_relu', block, {'conv_d0b-c'}, {'conv_d0b-c_relu'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d0c-1a', block, {'conv_d0b-c_relu'}, {'pool_d0c-1a'}, {});

% 2

block = dagnn.Conv('size', [3 3 64 128], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d1a-b', block, {'pool_d0c-1a'}, {'conv_d1a-b'}, {'conv_d1a-b_filter', 'conv_d1a-b_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d1a-b_relu', block, {'conv_d1a-b'}, {'conv_d1a-b_relu'}, {});

block = dagnn.Conv('size', [3 3 128 128], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d1b-c', block, {'conv_d1a-b_relu'}, {'conv_d1b-c'}, {'conv_d1b-c_filter', 'conv_d1b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d1b-c_relu', block, {'conv_d1b-c'}, {'conv_d1b-c_relu'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d1c-2a', block, {'conv_d1b-c_relu'}, {'pool_d1c-2a'}, {});

% 3

block = dagnn.Conv('size', [3 3 128 256], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d2a-b', block, {'pool_d1c-2a'}, {'conv_d2a-b'}, {'conv_d2a-b_filter', 'conv_d2a-b_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d2a-b_relu', block, {'conv_d2a-b'}, {'conv_d2a-b_relu'}, {});

block = dagnn.Conv('size', [3 3 256 256], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d2b-c', block, {'conv_d2a-b_relu'}, {'conv_d2b-c'}, {'conv_d2b-c_filter', 'conv_d2b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d2b-c_relu', block, {'conv_d2b-c'}, {'conv_d2b-c_relu'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d2c-3a', block, {'conv_d2b-c_relu'}, {'pool_d2c-3a'}, {});

% 4

block = dagnn.Conv('size', [3 3 256 512], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d3a-b', block, {'pool_d2c-3a'}, {'conv_d3a-b'}, {'conv_d3a-b_filter', 'conv_d3a-b_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d3a-b_relu', block, {'conv_d3a-b'}, {'conv_d3a-b_relu'}, {});

block = dagnn.Conv('size', [3 3 512 512], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d3b-c', block, {'conv_d3a-b_relu'}, {'conv_d3b-c'}, {'conv_d3b-c_filter', 'conv_d3b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d3b-c_relu', block, {'conv_d3b-c'}, {'conv_d3b-c_relu'}, {});

block = dagnn.DropOut('rate', 0.5);
net.addLayer('dropout_d3c', block, {'conv_d3b-c_relu'}, {'dropout_d3c'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d3c-4a', block, {'dropout_d3c'}, {'pool_d3c-4a'}, {});

% 5

block = dagnn.Conv('size', [3 3 512 1024], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d4a-b', block, {'pool_d3c-4a'}, {'conv_d4a-b'}, {'conv_d4a-b_filter', 'conv_d4a-b_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d4a-b_relu', block, {'conv_d4a-b'}, {'conv_d4a-b_relu'}, {});

block = dagnn.Conv('size', [3 3 1024 1024], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_d4b-c', block, {'conv_d4a-b_relu'}, {'conv_d4b-c'}, {'conv_d4b-c_filter', 'conv_d4b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d4b-c_relu', block, {'conv_d4b-c'}, {'conv_d4b-c_relu'}, {});

block = dagnn.DropOut('rate', 0.5);
net.addLayer('dropout_d4c', block, {'conv_d4b-c_relu'}, {'dropout_d4c'}, {});

% Deconv, Crop & Concat 3

block = dagnn.ConvTranspose('size', [2 2 1024 512], 'upsample', [2 2]);
net.addLayer('upconv_d4c_u3a', block, {'dropout_d4c'}, {'upconv_d4c_u3a'}, {});

block = dagnn.ReLU();
net.addLayer('upconv_d4c_u3a_relu', block, {'upconv_d4c_u3a'}, {'upconv_d4c_u3a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d3c-d3cc', block, {'dropout_d3c', 'upconv_d4c_u3a_relu'}, {'crop_d3c-d3cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d3cc_u3a-b', block, {'upconv_d4c_u3a_relu', 'crop_d3c-d3cc'}, {'concat_d3cc_u3a-b'}, {});

% 6

block = dagnn.Conv('size', [3 3 512 512], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u3b-c', block, {'concat_d3cc_u3a-b'}, {'conv_u3b-c'}, {'conv_u3b-c_filter', 'conv_u3b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u3a-b_relu', block, {'conv_u3a-b'}, {'conv_u3a-b_relu'}, {});

block = dagnn.Conv('size', [3 3 512 512], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u3b-c', block, {'conv_u3a-b_relu'}, {'conv_u3b-c'}, {'conv_u3b-c_filter', 'conv_u3b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u3b-c_relu', block, {'conv_u3b-c'}, {'conv_u3b-c_relu'}, {});

% Deconv, Crop & Concat 2

block = dagnn.ConvTranspose('size', [2 2 512 256], 'upsample', [2 2]);
net.addLayer('upconv_u3d_u2a', block, {'conv_u3b-c_relu'}, {'upconv_u3d_u2a'}, {});

block = dagnn.ReLU();
net.addLayer('upconv_u3d_u2a_relu', block, {'upconv_u3d_u2a'}, {'upconv_u3d_u2a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d2c-d2cc', block, {'conv_d2b-c_relu', 'upconv_u3d_u2a_relu'}, {'crop_d2c-d2cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d2cc_u2a-b', block, {'upconv_u3d_u2a_relu', 'crop_d2c-d2cc'}, {'concat_d2cc_u2a-b'}, {});

% 7

block = dagnn.Conv('size', [3 3 256 256], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u2b-c', block, {'concat_d2cc_u2a-b'}, {'conv_u2b-c'}, {'conv_u2b-c_filter', 'conv_u2b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u2b-c_relu', block, {'conv_u2b-c'}, {'conv_u2b-c_relu'}, {});

block = dagnn.Conv('size', [3 3 256 256], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u2c-d', block, {'conv_u2b-c_relu'}, {'conv_u2c-d'}, {'conv_u2c-d_filter', 'conv_u2c-d_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u2c-d_relu', block, {'conv_u2c-d'}, {'conv_u2c-d_relu'}, {});

% Deconv, Crop & Concat 1

block = dagnn.ConvTranspose('size', [2 2 256 128], 'upsample', [2 2]);
net.addLayer('upconv_u2d_u1a', block, {'conv_u2c-d_relu'}, {'upconv_u2d_u1a'}, {});

block = dagnn.ReLU();
net.addLayer('upconv_u2d_u1a_relu', block, {'upconv_u2d_u1a'}, {'upconv_u2d_u1a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d1c-d1cc', block, {'conv_d1b-c_relu', 'upconv_u2d_u1a_relu'}, {'crop_d1c-d1cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d1cc_u1a-b', block, {'upconv_u2d_u1a_relu', 'crop_d1c-d1cc'}, {'concat_d1cc_u1a-b'}, {});

% 8

block = dagnn.Conv('size', [3 3 128 128], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u1b-c', block, {'concat_d1cc_u1a-b'}, {'conv_u1b-c'}, {'conv_u1b-c_filter', 'conv_u1b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u1b-c_relu', block, {'conv_u1b-c'}, {'conv_u1b-c_relu'}, {});

block = dagnn.Conv('size', [3 3 128 128], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u1c-d', block, {'conv_u1b-c_relu'}, {'conv_u1c-d'}, {'conv_u1c-d_filter', 'conv_u1c-d_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u1c-d_relu', block, {'conv_u1c-d'}, {'conv_u1c-d_relu'}, {});

% Deconv, Crop & Concat 0

block = dagnn.ConvTranspose('size', [2 2 128 128], 'upsample', [2 2]);
net.addLayer('upconv_u1d_u0a', block, {'conv_u1c-d_relu'}, {'upconv_u1d_u0a'}, {});

block = dagnn.ReLU();
net.addLayer('upconv_u1d_u0a_relu', block, {'upconv_u1d_u0a'}, {'upconv_u1d_u0a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d0c-d0cc', block, {'conv_d0b-c_relu', 'upconv_u1d_u0a_relu'}, {'crop_d0c-d0cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d0cc_u0a-b', block, {'upconv_u1d_u0a_relu', 'crop_d0c-d0cc'}, {'concat_d0cc_u0a-b'}, {});

% 9

block = dagnn.Conv('size', [3 3 128 64], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u0b-c', block, {'concat_d0cc_u0a-b'}, {'conv_u0b-c'}, {'conv_u0b-c_filter', 'conv_u0b-c_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u0b-c_relu', block, {'conv_u0b-c'}, {'conv_u0b-c_relu'}, {});

block = dagnn.Conv('size', [3 3 64 64], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u0c-d', block, {'conv_u0b-c_relu'}, {'conv_u0c-d'}, {'conv_u0c-d_filter', 'conv_u0c-d_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u0c-d_relu', block, {'conv_u0c-d'}, {'conv_u0c-d_relu'}, {});

% Final Convolution

block = dagnn.Conv('size', [1 1 64 2], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u0d-score', block, {'conv_u0c-d_relu'}, {'conv_u0d-score'}, {'conv_u0d-score_filter', 'conv_u0d-score_bias'});


% Loss

net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'conv_u0d-score','label'}, 'objective') ;

% Initialize the network parameters
net.initParams();
