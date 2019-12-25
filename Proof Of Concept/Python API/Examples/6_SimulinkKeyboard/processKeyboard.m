function processKeyboard(actor, key, dt)
    
    %% Keys function
    %
    %   W       --->    Forward
    %   S       --->    Brake
    %   A       --->    Left
    %   D       --->    Right
    %   R       --->    Reverse gear
    %   F       --->    Forward gear
    %   E       --->    Enter auto control
    %   Q       --->    Quit auto control
    
    key = lower(key);
    
    if strcmp(key, 'timeout')
        return;
    elseif strcmp(key, char(0))
        control = actor.get_control();
        control.steer = 0;
        
        actor.apply_control(control);
    end
    
    % Change the drive mode
    if strcmp(key, 'e')
        actor.set_autopilot(true);

    elseif strcmp(key, 'q')
        actor.set_autopilot(false);
        
    % Throttle
    elseif strcmp(key, 'w')
        control = actor.get_control();
        control.throttle = control.throttle + 0.5 * dt;
        control.brake = 0;
        
        if control.throttle > 1
            control.throttle = 1;
        end
        
        actor.apply_control(control);
        
    elseif strcmp(key, 's')
        control = actor.get_control();
        control.throttle = 0;
        control.brake = 1;
        
        actor.apply_control(control);
    
    % Steer
    elseif strcmp(key, 'd')
        control = actor.get_control();
        
        control.steer = control.steer + 0.5 * dt;
        
        if control.steer > 1
            control.steer = 1;
        end
        
        actor.apply_control(control);
    
    elseif strcmp(key, 'a')
        control = actor.get_control();
        
        control.steer = control.steer - 0.5 * dt;
        
        if control.steer < -1
            control.steer = -1;
        end
        
        actor.apply_control(control);
    
    % Direction
    elseif strcmp(key, 'r')
        control = actor.get_control();
        control.reverse = true;
        
        actor.apply_control(control);
    elseif strcmp(key, 'f')
        control = actor.get_control();
        control.reverse = false;
        control.throttle = 0.5;
        
        actor.apply_control(control);        
    end
    
end