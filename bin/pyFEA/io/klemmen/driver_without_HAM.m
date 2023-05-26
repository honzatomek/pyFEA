%% Parameters.

% Define structural parameters.
structure;

% Initial velocity of Scheibe [rpm].
omega = 4800;

% Simulation end time [s].
tFinal = 0.300;

%% Parameters to customize the external force.

% Initial position of the clamping force on Scheibe according to the experiment
% (on the edge of the Scheibe, 11h 10min when viewed from the opposite side).
rEXT_SCH = 167.0E-3; phiEXT_SCH = 1.1345;  
par = defineClampingPoint(par, rEXT_SCH, phiEXT_SCH);

% Nominal clamping force [N].
extPar.FN = 500;

% Coefficient of dynamic friction between concrete probes and Scheibe [-].
extPar.mu = 0.51;

% Ramp-up, hold and ramp-down times for the clamping force.
extPar.tUp = 0.055;
extPar.tContact = 0.060;
extPar.tDown = 0.050;



% Solve the initial problem of clamping by concrete probes using fourth-order 
% Runge-Kutta method. The effect of external forces other than the clamping 
% force is neglected in this calculation.

%% Inital conditions.

rCOGx0 = 0;     % x-coordinate COG
rCOGy0 = 0;     % y-coordinate COG
phiCOGz0 = 0;   % angle about axis through COG

vCOGx0 = 0;     % x-velocity COG
vCOGy0 = 0;     % y-velocity COG
omCOGz0 = 0;    % angular velocity about axis through COG

omSCHz0 = -2*pi*omega/60; % angular velocity of Scheibe

%% Solver

% Pack initial conditions into a vector.
y0 = [rCOGx0; rCOGy0; phiCOGz0; vCOGx0; vCOGy0; omCOGz0; omSCHz0];
clear rCOGx0;
clear rCOGy0;
clear phiCOGz0;
clear vCOGx0;
clear vCOGy0;
clear omCOGz0;
clear omSCHz0;

% We have only the force on SCH here.
forceFcn = @(t, y) (fEXT(t, y, par, extPar));

% Solve.
[T, Y] = solver(forceFcn, par, 0.0, tFinal, y0);

% Unpack solution vector (first phase - clamping; without HAM).
[rCOGx, rCOGy, phiCOGz, vCOGx, vCOGy, omCOGz, omSCHz] = unpackVars(Y);
