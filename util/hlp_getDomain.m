% Helper function to the domain from patientName
function [domain, domainAlphabetic] = hlp_getDomain(patientName)
    if strncmpi(patientName, '010', 3)
        domain = 1;
        domainAlphabetic = 'A';
    elseif strncmpi(patientName, '070', 3)
        domain = 2;
        domainAlphabetic = 'B';
    elseif strncmpi(patientName, '080', 3)
        domain = 3;
        domainAlphabetic = 'C';
    elseif strncmpi(patientName, 'm_', 2)
        domain = 4;
        domainAlphabetic = 'D';
    else
        error('hlp_getDomain: Unknown domain!');
    end
end
