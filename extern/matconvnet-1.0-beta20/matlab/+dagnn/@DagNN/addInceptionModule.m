function addInceptionModule( net, modulename, inputlayername, outputlayername, varargin )
%INCEPTIONMODULE Add an Inception Module to your architecture as proposed in 
% "Szegedy et al.: Going deeper with Convolutions"
%
% INPUT:
%
%   modulename = a string containing the name of your module, i.e. 'icp1'
%   inputlayername = a string with the name of the layer that servers as
%                    input to the Inception Module
%   outputlayername = a string with the name of the Inception Module output
%                     layer
%
% Author: Christoph Baur <c.baur@tum.de>

    opts.inputChannels = 192;
    opts.reduction1 = 96; % Number of Filters
    opts.reduction2 = 16; % Number of Filters
    opts.conv1 = 64; %Number of Filters
    opts.conv2 = 128;
    opts.conv3 = 32;
    opts.conv4 = 32;
    opts = vl_argparse(opts, varargin, 'nonrecursive') ;

    block = dagnn.Conv('size', [1 1 opts.inputChannels opts.reduction1], 'hasBias', 1, 'pad', [0 0 0 0], 'stride', [1 1]);
    net.addLayer([modulename '_reduction1'], block, {inputlayername}, {[modulename '_reduction1']}, {[modulename '_reduction1_filter'], [modulename '_reduction1_bias']});
    block = dagnn.ReLU('useShortCircuit', 1, 'leak', 0);
    net.addLayer(['relu_' modulename '_reduction1'], block, {[modulename '_reduction1']}, {[modulename '_reduction1x']}, {});
    
    block = dagnn.Conv('size', [1 1 opts.inputChannels opts.reduction2], 'hasBias', 1, 'pad', [0 0 0 0], 'stride', [1 1]);
    net.addLayer([modulename '_reduction2'], block, {inputlayername}, {[modulename '_reduction2']}, {[modulename '_reduction2_filter'], [modulename '_reduction2_bias']});
    block = dagnn.ReLU('useShortCircuit', 1, 'leak', 0);
    net.addLayer(['relu_' modulename '_reduction2'], block, {[modulename '_reduction2']}, {[modulename '_reduction2x']}, {});
    
    block = dagnn.Pooling('method', 'max', 'poolSize', [3 3], 'pad', [1 1 1 1], 'stride', [1 1]);
    net.addLayer([modulename '_pool'], block, {inputlayername}, {[modulename '_pool']}, {});
    
    block = dagnn.Conv('size', [1 1 opts.inputChannels opts.conv1], 'hasBias', 1, 'pad', [0 0 0 0], 'stride', [1 1]);
    net.addLayer([modulename '_out0'], block, {inputlayername}, {[modulename '_out0']}, {[modulename '_out0_filter'], [modulename '_out0_bias']});
    block = dagnn.ReLU('useShortCircuit', 1, 'leak', 0);
    net.addLayer(['relu' modulename '_out0'], block, {[modulename '_out0']}, {[modulename '_out0x']}, {});
    
    block = dagnn.Conv('size', [3 3 opts.reduction1 opts.conv2], 'hasBias', 1, 'pad', [1 1 1 1], 'stride', [1 1]);
    net.addLayer([modulename '_out1'], block, {[modulename '_reduction1x']}, {[modulename '_out1']}, {[modulename '_out1_filter'], [modulename '_out1_bias']});
    block = dagnn.ReLU('useShortCircuit', 1, 'leak', 0);
    net.addLayer(['relu' modulename '_out1'], block, {[modulename '_out1']}, {[modulename '_out1x']}, {});
    
    block = dagnn.Conv('size', [5 5 opts.reduction2 opts.conv3], 'hasBias', 1, 'pad', [2 2 2 2], 'stride', [1 1]);
    net.addLayer([modulename '_out2'], block, {[modulename '_reduction2x']}, {[modulename '_out2']}, {[modulename '_out2_filter'], [modulename '_out2_bias']});
    block = dagnn.ReLU('useShortCircuit', 1, 'leak', 0);
    net.addLayer(['relu' modulename '_out2'], block, {[modulename '_out2']}, {[modulename '_out2x']}, {});
    
    block = dagnn.Conv('size', [1 1 opts.inputChannels opts.conv4], 'hasBias', 1, 'pad', [0 0 0 0], 'stride', [1 1]);
    net.addLayer([modulename '_out3'], block, {[modulename '_pool']}, {[modulename '_out3']}, {[modulename '_out3_filter'], [modulename '_out3_bias']});
    block = dagnn.ReLU('useShortCircuit', 1, 'leak', 0);
    net.addLayer(['relu' modulename '_out3'], block, {[modulename '_out3']}, {[modulename '_out3x']}, {});
    
    block = dagnn.Concat('dim', 3);
    net.addLayer(outputlayername, block, {[modulename '_out0x'], [modulename '_out1x'], [modulename '_out2x'], [modulename '_out3x']}, {outputlayername}, {});
end

