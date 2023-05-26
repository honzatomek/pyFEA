function [T, Y] = solver(forceFcn, par, tStart, tEnd, y0)

% Create properly parametrized function that returns vector of derivatives.
diffFcn = @(t, y) (fDiff(t, y, par, forceFcn));

% Solve.
[T, Y] = ode45(diffFcn, [tStart tEnd], y0);
