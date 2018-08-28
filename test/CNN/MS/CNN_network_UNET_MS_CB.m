% Creates the UNET network architecture for medical image segmentation in a fully convolutional way
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

block = dagnn.Conv('size', [3 3 numInputChannels 64], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d0ab', block, {'input'}, {'conv_d0ab'}, {'conv_d0ab_filter', 'conv_d0ab_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d0ab_relu', block, {'conv_d0ab'}, {'conv_d0ab_relu'}, {});

block = dagnn.Conv('size', [3 3 64 64], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d0bc', block, {'conv_d0ab_relu'}, {'conv_d0bc'}, {'conv_d0bc_filter', 'conv_d0bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d0bc_relu', block, {'conv_d0bc'}, {'conv_d0bc_relu'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d0c1a', block, {'conv_d0bc_relu'}, {'pool_d0c1a'}, {});

% 2

block = dagnn.Conv('size', [3 3 64 128], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d1ab', block, {'pool_d0c1a'}, {'conv_d1ab'}, {'conv_d1ab_filter', 'conv_d1ab_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d1ab_relu', block, {'conv_d1ab'}, {'conv_d1ab_relu'}, {});

block = dagnn.Conv('size', [3 3 128 128], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d1bc', block, {'conv_d1ab_relu'}, {'conv_d1bc'}, {'conv_d1bc_filter', 'conv_d1bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d1bc_relu', block, {'conv_d1bc'}, {'conv_d1bc_relu'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d1c2a', block, {'conv_d1bc_relu'}, {'pool_d1c2a'}, {});

% 3

block = dagnn.Conv('size', [3 3 128 256], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d2ab', block, {'pool_d1c2a'}, {'conv_d2ab'}, {'conv_d2ab_filter', 'conv_d2ab_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d2ab_relu', block, {'conv_d2ab'}, {'conv_d2ab_relu'}, {});

block = dagnn.Conv('size', [3 3 256 256], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d2bc', block, {'conv_d2ab_relu'}, {'conv_d2bc'}, {'conv_d2bc_filter', 'conv_d2bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d2bc_relu', block, {'conv_d2bc'}, {'conv_d2bc_relu'}, {});

block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
net.addLayer('pool_d2c3a', block, {'conv_d2bc_relu'}, {'pool_d2c3a'}, {});

% 4

block = dagnn.Conv('size', [3 3 256 512], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d3ab', block, {'pool_d2c3a'}, {'conv_d3ab'}, {'conv_d3ab_filter', 'conv_d3ab_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d3ab_relu', block, {'conv_d3ab'}, {'conv_d3ab_relu'}, {});

block = dagnn.Conv('size', [3 3 512 512], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_d3bc', block, {'conv_d3ab_relu'}, {'conv_d3bc'}, {'conv_d3bc_filter', 'conv_d3bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_d3bc_relu', block, {'conv_d3bc'}, {'conv_d3bc_relu'}, {});

block = dagnn.DropOut('rate', 0.5);
net.addLayer('dropout_d3c', block, {'conv_d3bc_relu'}, {'dropout_d3c'}, {});

% block = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', [0 0 0 0], 'stride', [2 2]);
% net.addLayer('pool_d3c4a', block, {'dropout_d3c'}, {'pool_d3c4a'}, {});

% 5

% block = dagnn.Conv('size', [3 3 512 1024], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
% net.addLayer('conv_d4ab', block, {'pool_d3c4a'}, {'conv_d4ab'}, {'conv_d4ab_filter', 'conv_d4ab_bias'});
% 
% block = dagnn.ReLU();
% net.addLayer('conv_d4ab_relu', block, {'conv_d4ab'}, {'conv_d4ab_relu'}, {});
% 
% block = dagnn.Conv('size', [3 3 1024 1024], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
% net.addLayer('conv_d4bc', block, {'conv_d4ab_relu'}, {'conv_d4bc'}, {'conv_d4bc_filter', 'conv_d4bc_bias'});
% 
% block = dagnn.ReLU();
% net.addLayer('conv_d4bc_relu', block, {'conv_d4bc'}, {'conv_d4bc_relu'}, {});
% 
% block = dagnn.DropOut('rate', 0.5);
% net.addLayer('dropout_d4c', block, {'conv_d4bc_relu'}, {'dropout_d4c'}, {});

% Deconv, Crop & Concat 3

% block = dagnn.ConvTranspose('size', [2 2 1024 512], 'upsample', [2 2]);
% net.addLayer('upconv_d4c_u3a', block, {'dropout_d4c'}, {'upconv_d4c_u3a'}, {'upconv_d4c_u3a_filter', 'upconv_d4c_u3a_bias'});
% 
% block = dagnn.ReLU();
% net.addLayer('upconv_d4c_u3a_relu', block, {'upconv_d4c_u3a'}, {'upconv_d4c_u3a_relu'}, {});
% 
% block = dagnn.Crop();
% net.addLayer('crop_d3cd3cc', block, {'dropout_d3c', 'upconv_d4c_u3a_relu'}, {'crop_d3cd3cc'}, {});
% 
% block = dagnn.Concat();
% net.addLayer('concat_d3cc_u3ab', block, {'upconv_d4c_u3a_relu', 'crop_d3cd3cc'}, {'concat_d3cc_u3ab'}, {});

% 6

% block = dagnn.Conv('size', [3 3 512 512], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
% net.addLayer('conv_u3ab', block, {'concat_d3cc_u3ab'}, {'conv_u3ab'}, {'conv_u3ab_filter', 'conv_u3ab_bias'});
% 
% block = dagnn.ReLU();
% net.addLayer('conv_u3ab_relu', block, {'conv_u3ab'}, {'conv_u3ab_relu'}, {});
% 
% block = dagnn.Conv('size', [3 3 512 512], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
% net.addLayer('conv_u3bc', block, {'conv_u3ab_relu'}, {'conv_u3bc'}, {'conv_u3bc_filter', 'conv_u3bc_bias'});
% 
% block = dagnn.ReLU();
% net.addLayer('conv_u3bc_relu', block, {'conv_u3bc'}, {'conv_u3bc_relu'}, {});

% Deconv, Crop & Concat 2

block = dagnn.ConvTranspose('size', [2 2 256 512], 'upsample', [2 2]);
net.addLayer('upconv_u3d_u2a', block, {'dropout_d3c'}, {'upconv_u3d_u2a'}, {'upconv_u3d_u2a_filter', 'upconv_u3d_u2a_bias'});

block = dagnn.ReLU();
net.addLayer('upconv_u3d_u2a_relu', block, {'upconv_u3d_u2a'}, {'upconv_u3d_u2a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d2cd2cc', block, {'conv_d2bc_relu', 'upconv_u3d_u2a_relu'}, {'crop_d2cd2cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d2cc_u2ab', block, {'upconv_u3d_u2a_relu', 'crop_d2cd2cc'}, {'concat_d2cc_u2ab'}, {});

% 7

block = dagnn.Conv('size', [3 3 512 256], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_u2bc', block, {'concat_d2cc_u2ab'}, {'conv_u2bc'}, {'conv_u2bc_filter', 'conv_u2bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u2bc_relu', block, {'conv_u2bc'}, {'conv_u2bc_relu'}, {});

block = dagnn.Conv('size', [3 3 256 256], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_u2cd', block, {'conv_u2bc_relu'}, {'conv_u2cd'}, {'conv_u2cd_filter', 'conv_u2cd_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u2cd_relu', block, {'conv_u2cd'}, {'conv_u2cd_relu'}, {});

% Deconv, Crop & Concat 1

block = dagnn.ConvTranspose('size', [2 2 128 256], 'upsample', [2 2]);
net.addLayer('upconv_u2d_u1a', block, {'conv_u2cd_relu'}, {'upconv_u2d_u1a'}, {'upconv_u2d_u1a_filter', 'upconv_u2d_u1a_bias'});

block = dagnn.ReLU();
net.addLayer('upconv_u2d_u1a_relu', block, {'upconv_u2d_u1a'}, {'upconv_u2d_u1a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d1cd1cc', block, {'conv_d1bc_relu', 'upconv_u2d_u1a_relu'}, {'crop_d1cd1cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d1cc_u1ab', block, {'upconv_u2d_u1a_relu', 'crop_d1cd1cc'}, {'concat_d1cc_u1ab'}, {});

% 8

block = dagnn.Conv('size', [3 3 256 128], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_u1bc', block, {'concat_d1cc_u1ab'}, {'conv_u1bc'}, {'conv_u1bc_filter', 'conv_u1bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u1bc_relu', block, {'conv_u1bc'}, {'conv_u1bc_relu'}, {});

block = dagnn.Conv('size', [3 3 128 128], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_u1cd', block, {'conv_u1bc_relu'}, {'conv_u1cd'}, {'conv_u1cd_filter', 'conv_u1cd_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u1cd_relu', block, {'conv_u1cd'}, {'conv_u1cd_relu'}, {});

% Deconv, Crop & Concat 0

block = dagnn.ConvTranspose('size', [2 2 128 128], 'upsample', [2 2]);
net.addLayer('upconv_u1d_u0a', block, {'conv_u1cd_relu'}, {'upconv_u1d_u0a'}, {'upconv_u1d_u0a_filter', 'upconv_u1d_u0a_bias'});

block = dagnn.ReLU();
net.addLayer('upconv_u1d_u0a_relu', block, {'upconv_u1d_u0a'}, {'upconv_u1d_u0a_relu'}, {});

block = dagnn.Crop();
net.addLayer('crop_d0cd0cc', block, {'conv_d0bc_relu', 'upconv_u1d_u0a_relu'}, {'crop_d0cd0cc'}, {});

block = dagnn.Concat();
net.addLayer('concat_d0cc_u0ab', block, {'upconv_u1d_u0a_relu', 'crop_d0cd0cc'}, {'concat_d0cc_u0ab'}, {});

% 9

block = dagnn.Conv('size', [3 3 192 64], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_u0bc', block, {'concat_d0cc_u0ab'}, {'conv_u0bc'}, {'conv_u0bc_filter', 'conv_u0bc_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u0bc_relu', block, {'conv_u0bc'}, {'conv_u0bc_relu'}, {});

block = dagnn.Conv('size', [3 3 64 64], 'hasBias', 1, 'pad', [1 1], 'stride', [1 1]);
net.addLayer('conv_u0cd', block, {'conv_u0bc_relu'}, {'conv_u0cd'}, {'conv_u0cd_filter', 'conv_u0cd_bias'});

block = dagnn.ReLU();
net.addLayer('conv_u0cd_relu', block, {'conv_u0cd'}, {'conv_u0cd_relu'}, {});

% Final Convolution

block = dagnn.Conv('size', [1 1 64 2], 'hasBias', 1, 'pad', [0 0], 'stride', [1 1]);
net.addLayer('conv_u0d_score', block, {'conv_u0cd_relu'}, {'conv_u0d_score'}, {'conv_u0d_score_filter', 'conv_u0d_score_bias'});

% Loss

net.addLayer('prob', dagnn.SoftMax(), {'conv_u0d_score'}, {'prediction'}, {});
%net.addLayer('prob', dagnn.Sigmoid(), {'conv_u0d_score'}, {'prediction'}, {});

net.addLayer('label_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label'}, {'label_ignoreBG'}, {});

net.addLayer('loss', dagnn.FbetaLoss('updateBeta', true, 'beta_lr', 100), {'prediction','label_ignoreBG'}, 'objective', {'beta'}) ;

% Initialize the network parameters
net.initParams();
