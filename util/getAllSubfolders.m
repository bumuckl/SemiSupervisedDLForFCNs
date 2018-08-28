% Get all files inside a directory. Filter them by fileExtensions and
% obtain the fullpath if desired

function subfolders = getAllSubfolders(dirName, appendFullPath)

  dirWithSubFolders = dir(dirName);
  dirIndex = [dirWithSubFolders.isdir];  %# Find the index for directories
  subDirs = {dirWithSubFolders(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
	if appendFullPath
    subDirs = cellfun(@(x) fullfile(dirName, x),...  %# Prepend path to files
                       subDirs,'UniformOutput',false);
	end																				
  subfolders = subDirs(validIndex);
end