function meas = helperCvmeasCuboid(states,varargin)
    % This is a helper function and may be removed in a future release.
    % This function is the measurement model for constant velocity state.
    
    % Copyright 2019 The MathWorks, Inc.
    
    % Get position, dimension and yaw from state and use the
    % lidarModel to obtain measurements.
    pos = states([1 3 5],:);
    dim = states([8 9 10],:);
    yaw = states(7,:);
    meas = helperLidarModel(pos,dim,yaw);
end