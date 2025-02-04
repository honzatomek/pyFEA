function plotFrame(t, par, rCOGx, rCOGy, phiCOGz)

% Initialize plotting.
clf;
hold on;
axis equal
title(sprintf('t = %6.4fs', t));

% Plot CS0
line([-0.9 0.9], [0 0]);
line([0 0], [-0.9 0.9]);

% Plot COG.
% scatter(rCOGx, rCOGy, 50, 'black', 'filled');
scatter(rCOGx, rCOGy, 50, 'k', 'filled');

% Plot GRO and GRS.
% plotPointFcn = @(r, phi) (plotPoint(rCOGx, rCOGy, phiCOGz, 'blue', r, phi));
plotPointFcn = @(r, phi) (plotPoint(rCOGx, rCOGy, phiCOGz, 'b', r, phi));
plotPointFcn(par.rGRO_COG, par.phiGRO_COG)
plotPointFcn(par.rGRS_COG, par.phiGRS_COG)

% Plot KUW, RTS and AWL.
% plotPointFcn = @(r, phi) (plotPoint(rCOGx, rCOGy, phiCOGz, 'red', r, phi));
plotPointFcn = @(r, phi) (plotPoint(rCOGx, rCOGy, phiCOGz, 'r', r, phi));
plotPointFcn(par.rKUW_COG, par.phiKUW_COG)
plotPointFcn(par.rRTS_COG, par.phiRTS_COG)
plotPointFcn(par.rAWL_COG, par.phiAWL_COG)

% Plot SCH.
% plotPointFcn = @(r, phi) (plotPoint(rCOGx, rCOGy, phiCOGz, 'green', r, phi));
plotPointFcn = @(r, phi) (plotPoint(rCOGx, rCOGy, phiCOGz, 'g', r, phi));
plotPointFcn(par.rSCH_COG, par.phiSCH_COG)
xCenter = rCOGx + par.rSCH_COG*cos(par.phiSCH_COG + phiCOGz);
yCenter = rCOGy + par.rSCH_COG*sin(par.phiSCH_COG + phiCOGz);
plotCircle(xCenter, yCenter, 167.0E-3);

% Plot Proband.

xLL = -0.5; yLL = -1.0;
xRL = -0.3; yRL = -1.0;
xBD = -0.4; yBD = -0.3;
xBT = -0.4; yBT = +0.5;

xLH = rCOGx + par.rGRO_COG*cos(par.phiGRO_COG + phiCOGz);
yLH = rCOGy + par.rGRO_COG*sin(par.phiGRO_COG + phiCOGz);

xRH = rCOGx + par.rGRS_COG*cos(par.phiGRS_COG + phiCOGz);
yRH = rCOGy + par.rGRS_COG*sin(par.phiGRS_COG + phiCOGz);

rH  = 100.0E-3;

line([xLL xBD], [yLL yBD]);
line([xRL xBD], [yRL yBD]);

line([xLH xBT], [yLH yBT - rH]);
line([xRH xBT], [yRH yBT - rH]);

line([xBD xBT], [yBD yBT]);

plotCircle(xBT, yBT + rH, rH)


% External force (not implemented yet)!
% Fx = FEXT(1);
% Fy = FEXT(2);
% dx = points(5, 1).*cos(points(5, 2));
% dy = points(5, 1).*sin(points(5, 2));
% quiver(dx, dy, Fx/1000, Fy/1000);
