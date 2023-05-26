function [] = plotPoint(rCOGx, rCOGy, phiCOGz, color, r, phi)

x = rCOGx + r*cos(phi + phiCOGz);
y = rCOGy + r*sin(phi + phiCOGz);

line([rCOGx x], [rCOGy y]);
scatter(x, y, 20, color, 'filled');
