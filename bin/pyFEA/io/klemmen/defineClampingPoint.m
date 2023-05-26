function [par] = defineClampingPoint(par, rEXT_SCH, phiEXT_SCH)

% Takes the struture definition array and adds the definition of the clamping 
% point in various coordinate systems.

% Initial position of the clamping force with respect to SCH.
par.rEXT_SCH = rEXT_SCH; 
par.phiEXT_SCH = phiEXT_SCH;

% (Fixed) Position of external force in CS0 (assumes COG in origin at t = 0 so 
% that rCOG_CS0 is zero and can be omitted from the formula below).
rEXT_CS0x = par.rSCH_COG*cos(par.phiSCH_COG) + par.rEXT_SCH*cos(par.phiEXT_SCH);
rEXT_CS0y = par.rSCH_COG*sin(par.phiSCH_COG) + par.rEXT_SCH*sin(par.phiEXT_SCH);

% Recalculate the position of the external force to polar coordinates.
par.rEXT_CS0   = hypot(rEXT_CS0x, rEXT_CS0y); 
par.phiEXT_CS0 = atan2(rEXT_CS0y, rEXT_CS0x);
