function states = helperConstvelCuboid(states,v,dT)
    % This is a helper function and may be removed in a future release.

    % Copyright 2019 The MathWorks, Inc.
    
    % state is defined as [x;vx;y;vy;z;vz;yaw;l;w;h];
    % v is process noise defined as [ax;ay;az;omega];
    kinematicStates = states(1:6,:);
    if ~isscalar(v)
        kinematicStatesDt = constvel(kinematicStates,v(1:3,:),dT);
    else
        kinematicStatesDt = constvel(kinematicStates,dT);
    end
    % yaw is Constant
    yaw = states(7,:);
    if ~isscalar(v)
        yaw = yaw + v(4,:)*dT;
    end
    shapeStates = states(8:10,:);
    
    % Preserve size and data type for codegen.
    states(:) = [kinematicStatesDt;yaw;shapeStates];
end
