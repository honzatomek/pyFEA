%% Subcomponent definitions.

% Initial position of the machine in the experiment (rotated by 9° clockwise 
% with respect to CS0).
par.phiCOGz0 = -0.1559;

% Definition of subcomponents.
% COG   Center of gravity of the whole machine.
% KUW   Kurbelwelle and attached parts rotating with it.
% SCH   Diamantscheibe
% AWL   Abtriebswelle
% RTS   Rest of the machine.

% Position of subcomponent COGs with respect to the COG of the machine [m].
par.rKUW_COG =  96.0E-3;	par.phiKUW_COG = -2.792 + par.phiCOGz0; 
par.rSCH_COG = 225.0E-3;	par.phiSCH_COG =  0.137 + par.phiCOGz0; 
par.rAWL_COG = 225.0E-3;	par.phiAWL_COG =  0.137 + par.phiCOGz0; 
par.rRTS_COG =  43.7E-3;	par.phiRTS_COG = -3.073 + par.phiCOGz0; 
par.rGRO_COG = 174.5E-3;	par.phiGRO_COG =  1.571 + par.phiCOGz0;
par.rGRS_COG = 285.3E-3;	par.phiGRS_COG =  2.711 + par.phiCOGz0;

% Define subcomponent masses [kg] and moments of inertia (with respect to COG 
% of the subcomponent and rotation about z-axis) [kg.m^2].
par.mKUW = 1337.2E-3;       par.JKUW =   1180893.9E-9;
par.mSCH = 1668.0E-3;       par.JSCH =  25634000.0E-9;
par.mAWL =  643.8E-3;       par.JAWL =    601657.3E-9;
par.mRTS = 9065.9E-3;       par.JRTS = 257954230.0E-9;

% Total mass of the machine.
par.mTOT = par.mRTS + par.mKUW + par.mAWL + par.mSCH; 

% Moment of inertia of independently rotating parts (KUW + SCH).
par.JROT = par.JKUW + par.JSCH; 

% Moment of inertia of the machine with respect to rotation about COG.
par.JTOT = par.mKUW*par.rKUW_COG^2 + ...
           par.mSCH*par.rSCH_COG^2 + ...
           par.mAWL*par.rAWL_COG^2 + ...
           par.mRTS*par.rRTS_COG^2 + par.JAWL + par.JRTS;
