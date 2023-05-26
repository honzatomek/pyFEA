function [rCOGx, rCOGy, phiCOGz, vCOGx, vCOGy, omCOGz, omSCHz] = unpackVars(y)

    if size(y, 2) == 1
        y = y.';
    end % if

    % Unpack the vector of unknowns into individual variables.
    rCOGx =   y(:, 1);    % x-position of COG
    rCOGy =   y(:, 2);    % y-position of COG
    phiCOGz = y(:, 3);    % angle about axis through COG
    vCOGx =   y(:, 4);    % x-velocity COG
    vCOGy =   y(:, 5);    % y-velocity COG
    omCOGz =  y(:, 6);    % angular velocity about axis through COG
    omSCHz =  y(:, 7);    % angular velocity of Schiebe
