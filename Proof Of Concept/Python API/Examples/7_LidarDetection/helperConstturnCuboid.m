function state = helperConstturnCuboid(state,v,dT)
    % This is a helper function and may be removed in a future release.

    % Copyright 2019 The MathWorks, Inc.

    % convention for the state is: [x;vx;y;vy;w;z;vz;theta;l;w;h]; w is
    % degrees/second. yaw is degrees. v is defined as process noise using the
    % following convention: [ax;ay;alpha;az];

    % Predict the constant turn state using constturn model.
    constturnState = state(1:7,:);
    if ~isscalar(v)
        constturnStateDt = constturn(constturnState,v(1:4,:),dT);
    else
        constturnStateDt = constturn(constturnState,dT);
    end

    if ~isscalar(v)
        theta = state(7,:) + state(8,:)*dT + v(3,:).*dT^2/2;
    else
        theta = state(7,:) + state(8,:)*dT;
    end

    shapeState = state(9:end,:);

    % Preserve size and data type for codegen using colon operator.
    state(:) = [constturnStateDt;theta;shapeState];
end