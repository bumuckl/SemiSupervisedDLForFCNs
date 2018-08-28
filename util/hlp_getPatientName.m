% Helper function to obtain the patientName from a Volume path of MSSEG
function patientName = hlp_getPatientName(filename)
    [pathstr,name,ext] = fileparts(filename);
    if ispc
        parts = strsplit(pathstr, '\');
    else
        parts = strsplit(pathstr, '/');
    end
    patientName = parts{end};
end