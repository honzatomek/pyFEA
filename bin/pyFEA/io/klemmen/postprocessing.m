%% Postprocessing

% Load structure; assumes that the equations have been solved and the kinematic
% quantities are in the workspace.
structure;

% Drop of angular velocity of SCH [rpm].
deltaOmSCHz = 60*abs(omSCHz(end) - omSCHz(1))/(2*pi);

% Kickback angle [deg]; this is the angle between x-axis of the global
% coordinate system and the line connecting (1) the clamping position 
% on Scheibe with respect to CS0 and (2) initial position of the back Handgriff 
% with respect to CS0.

% Inital position of back handle.
rGRS_CS0x0 = rCOGx(1) + par.rGRS_COG*cos(par.phiGRS_COG);
rGRS_CS0y0 = rCOGy(1) + par.rGRS_COG*sin(par.phiGRS_COG);

% Position of clamping point on Scheibe at time t.
rCLA_CS0x = rCOGx + par.rEXT_CS0*cos(par.phiEXT_CS0 + phiCOGz);
rCLA_CS0y = rCOGy + par.rEXT_CS0*sin(par.phiEXT_CS0 + phiCOGz);

% Kickback angle.
dx = rCLA_CS0x - rGRS_CS0x0;
dy = rCLA_CS0y - rGRS_CS0y0;
kickback = 180*atan2(dy, dx)/pi;

% In experiment we also measure maximum speed of the point AWL, i.e. the speed
% with which the AWL goes against the unfortunate user.

% Velocity of AWL.
vAWL_CS0x = vCOGx - par.rAWL_COG*omCOGz.*sin(par.phiAWL_COG + phiCOGz);
vAWL_CS0y = vCOGy + par.rAWL_COG*omCOGz.*cos(par.phiAWL_COG + phiCOGz);

% Speed of AWL.
vAWL = hypot(vAWL_CS0x, vAWL_CS0y);

% Trajectory of COG (stored in rCOGx and rCOGy) and AWL (needs to be calculated).

% Position of AWL.
rAWL_CS0x = rCOGx + par.rAWL_COG*cos(par.phiAWL_COG + phiCOGz);
rAWL_CS0y = rCOGy + par.rAWL_COG*sin(par.phiAWL_COG + phiCOGz);
