function [FCOGx, FCOGy, MCOGz, MSCHz] = fEXT(t, y, par, extPar)
% Calculate the forces and moments caused by the clamping force.

% Define structural parameters.
structure;

% Unpack current solution vector.
[rCOGx, rCOGy, phiCOGz, vCOGx, vCOGy, omCOGz, omSCHz] = unpackVars(y);

% FN is the normal clamping force FN [from experiment]
% mu is the dynamic friction coefficient between steel and concrete [fitted] 
% tUp is time when the normal force is gradually ramped up [fitted]
% tConstant is time when the normal force is fully applied [fitted]
% tDown is time when the normal force is gradually ramped up [fitted]

% Time-dependent function describing ramp-up and ramp-down: first part is half
% period on sine squared [from 0 to pi/2], then constant and then half period 
% of sine squared [from pi/2 to pi].

if (t < extPar.tUp) % Ramp-up.
    f = sin((pi*t)/(2*extPar.tUp)).^2;
elseif (t < extPar.tUp + extPar.tContact) % Constant phase.
    f = 1;  
elseif (t < extPar.tUp + extPar.tContact + extPar.tDown) % Ramp-down.
    f = cos((pi*(t - (extPar.tUp + extPar.tContact)))/(2*extPar.tDown)).^2;
else % Clamping fully released.
    f = 0;
end

% Calculate the magnitude of the clamping force.
FEXT = extPar.mu*f*extPar.FN;

% We need to calculate the position of the clamping force with respect to SCH 
% because the components of this force depend on this position.

% Step 1: position of SCH with respect to COG.
rSCH_COGx = par.rSCH_COG*cos(par.phiSCH_COG + phiCOGz);
rSCH_COGy = par.rSCH_COG*sin(par.phiSCH_COG + phiCOGz);

% Step 2: position of EXT with respect to SCH.
rEXT_SCHx = par.rEXT_CS0*cos(par.phiEXT_CS0) - (rCOGx + rSCH_COGx);
rEXT_SCHy = par.rEXT_CS0*sin(par.phiEXT_CS0) - (rCOGy + rSCH_COGy);

rEXT_SCH = hypot(rEXT_SCHx, rEXT_SCHy);

% Step 3: external force.
FEXTx = FEXT*(-rEXT_SCHy/rEXT_SCH);
FEXTy = FEXT*(+rEXT_SCHx/rEXT_SCH);

% Create output variables.
FCOGx = FEXTx;
FCOGy = FEXTy;
MCOGz = rSCH_COGx*FEXTy - rSCH_COGy*FEXTx;
MSCHz = rEXT_SCHx*FEXTy - rEXT_SCHy*FEXTx;
