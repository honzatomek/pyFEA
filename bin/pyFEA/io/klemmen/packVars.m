function [y] = packVars(rCOGx, rCOGy, phiCOGz, vCOGx, vCOGy, omCOGz, omSCHz)

    % Pack unknowns into a single vector.
    y = [rCOGx; rCOGy; phiCOGz; vCOGx; vCOGy; omCOGz; omSCHz];
