function meas = helperCtmeasCuboid(state,varargin)
    % This is a helper function and may be removed in a future release.
    % This function is the measurement model for constant turn-rate state.
    
    % Copyright 2019 The MathWorks, Inc.
    
    % Get position, dimension and yaw from state and use the
    % lidarModel to obtain measurements.
    pos = state([1 3 6],:);
    dim = state([9 10 11],:);
    yaw = state(8,:);
    meas = helperLidarModel(pos,dim,yaw);
end