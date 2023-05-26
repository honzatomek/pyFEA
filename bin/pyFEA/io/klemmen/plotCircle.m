function [] = plotCircle(xCenter, yCenter, radius)

NPOINTS = 37;

tau = linspace(0, 2*pi, NPOINTS);
xCirc = ones(size(tau)).*(xCenter + radius*cos(tau));
yCirc = ones(size(tau)).*(yCenter + radius*sin(tau));
line(xCirc, yCirc)
