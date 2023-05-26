function [FCOGx, FCOGy, MCOGz, MSCHz] = fHAM(t, y, par, hamPar)
% Calculate the forces and moments caused by the Hand-Arm-System (variant
% according to the ANSI norm for Chainsaw kickbacks without chain brake).

% BEWARE!!! Everything has to be converted into imperial units and back!

% Effective radius of the Hand-Arm-System.
rHAM = (par.rGRO_COG + par.rGRS_COG)/0.0254;        % [in]

% Maximum moment transferrable by the Hand-Arm-System.
MMAX = 15*rHAM;                             % [lbf*in]

% Recalculate energies into imperial units.
E_TRx_EXP = hamPar.E_TRx_EXP/0.1129848290276167;           % [lb*in]
E_ROT_EXP = hamPar.E_ROT_EXP/0.1129848290276167;           % [lb*in]

% Transform experimental kickback energies into those suitable for HAM.
E_TRx = E_TRx_EXP;
E_TRy = min(E_TRx_EXP, E_ROT_EXP/3);
E_ROT = E_ROT_EXP - E_TRy;

WTOT = 9.80665*par.mTOT/4.4482216152605;        % [lbf]
JTOT = par.JTOT/0.1130;                         % [lb*in*s^2]

J1 = 25.554*(E_ROT^0.25)*sqrt(rHAM*JTOT);
J2 = -3.614*((WTOT*E_TRx)^0.25) + 6.012;
J3 = 0.0713*J1 + 3.047*WTOT - 5.215*sqrt(E_TRy) + 2.989;

K1 = 0.1070;
K3 = 0.1128;
O1 = 0.0000;
O3 = 0.32764*WTOT + 2.063;

% Calculation of effective force (x-component).
if (t < 0.1800)
    FCOGx = J2;
else
    if (J2 > 0)
        FCOGx = max(0, J2 - 200*t + 36);
    else
        FCOGx = min(0, J2 + 200*t - 36);
    end
end

% Calculation of effective force (y-component).
if (t < 0.1128)
    FCOGy = -(O3 - J3)*((t/K3)^2 - 2*(t/K3)) + J3;
elseif (t < 0.1500)
    FCOGy = O3;
else
    FCOGy = max(0, O3 - 200*t + 30);
end
    

% Calculation of effective moment.
if (t < 0.1070)
    MCOGz = -(O1 - J1)*((t/K1)^2 - 2*(t/K1)) + J1;
elseif (t < 0.1600)
    MCOGz = O1;
else
    MCOGz = min(MMAX, O1 + 1741*t - 278.56);
end

% Recalculate back to SI units [conversion factors from Wikipedia]. 
% NOTE: direction of FCOGx and Mz is reversed since the orientation of the
% machine is different than in the norm (Scheibe on the right side instead 
% on the left side.
FCOGx = -4.4482216152605*FCOGx;
FCOGy =  4.4482216152605*(FCOGy - WTOT);
MCOGz = -0.1129848290276*MCOGz;
MSCHz = 0;
