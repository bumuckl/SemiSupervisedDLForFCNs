% Helper function to obtain the patientName from a Volume path of MSSEG
function patientIdx = hlp_getPatientIdx(patientPaths, patientName)
    patientNames = {};
    for i=1:length(patientPaths)
        [pathstr,name,ext] = fileparts(patientPaths{i});
        if ispc
            parts = strsplit(pathstr, '\');
        else
            parts = strsplit(pathstr, '/');
        end
        patientNames{end+1} = parts{end};
    end
    
    patientIdx = find(strcmp(patientNames, patientName));
end