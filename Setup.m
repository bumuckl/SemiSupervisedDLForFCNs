%% Setup.m
% Add various folders to the paths and initialize Constants
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

this = struct;
this.filename = mfilename('fullpath');
[this.path, this.name, this.ext] = fileparts(this.filename);

% Add Paths
addpath(this.path);
addpath([this.path '/util']);
addpath([this.path '/extern']);
addpath([this.path '/extern/tSNE_matlab']);
addpath([this.path '/extern/HTMLReport']);
addpath([this.path '/extern/matconvnet-1.0-beta20/matlab']);
addpath('path_to_spm12');

% Run child setups
vl_setupnn;

% Set global variables
global DATAPATH;
global IMDBPATH;
global MSSEGDATAPATH;
global MSKRIDATAPATH;
global ROBEXBATPATH;
DATAPATH = [this.path '/../../data/'];
MSSEGDATAPATH = [this.path '/../../data/MSSEG'];
MSKRIDATAPATH = [this.path '/../../data/MSKRI'];
IMDBPATH = [this.path '/../../data/IMDB/'];
ROBEXBATPATH = '~\Downloads\ROBEXv12.win\ROBEX\runROBEX.bat';