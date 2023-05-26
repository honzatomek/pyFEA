% Simple animation of the movement.

for j = 1:length(T)
    
    % Draw single frame
    plotFrame(T(j), par, rCOGx(j), rCOGy(j), phiCOGz(j));
    pause(0.01);
    
    % Output to png-files; this can be used to create animated gif.
     frame = getframe(gcf);
     imwrite(frame.cdata, sprintf('frame%03d.png', j),'png');
    
end % for
