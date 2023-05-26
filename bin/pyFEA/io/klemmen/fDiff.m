function [yd] = fDiff(t, y, par, forceFcn)

% Function passed to the MATLAB differential equation solvers; calculate
% derivatives of the solution vector based on the current time and actual 
% solution vector.

% Unpack solution vector.
[rCOGx, rCOGy, phiCOGz, vCOGx, vCOGy, omCOGz, omSCHz] = unpackVars(y);

% Calculate the external forces.
[FCOGx, FCOGy, MCOGz, MSCHz] = forceFcn(t, y);

% Calculate derivatives of the solution vector.
rCOGxD = vCOGx;      
rCOGyD = vCOGy;      
phiCOGzD = omCOGz;   

vCOGxD = FCOGx/par.mTOT; 
vCOGyD = FCOGy/par.mTOT; 
omCOGzD = MCOGz/par.JTOT; 
omSCHzD = MSCHz/par.JROT; 

% Pack the derivatives.
yd = packVars(rCOGxD, rCOGyD, phiCOGzD, vCOGxD, vCOGyD, omCOGzD, omSCHzD);
