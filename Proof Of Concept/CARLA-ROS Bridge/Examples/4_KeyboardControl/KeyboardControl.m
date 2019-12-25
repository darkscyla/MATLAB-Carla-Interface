classdef KeyboardControl < matlab.System & matlab.system.mixin.Propagates
    % Carla Ros Bridge Keyboard Control Block
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties
        
    end

    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)
          
        key;
        keyboard_handle;
        time_last_update = 0;
    end

    methods(Access = protected)
        function [steer,throttle,reverse_gear,brake,manualcontrol,autopilot] = setupImpl(obj,steer_prev,throttle_prev,reverse_gear_prev,brake_prev,manualcontrol_prev,autopilot_prev)
            % Perform one-time calculations, such as computing constant
            
            % Handle keyboard
            callstr = 'set(gcbf, ''Userdata'', get(gcbf, ''Currentkey'')) ; uiresume ' ;
            obj.keyboard_handle = figure( 'name', 'Press a key', ...
                                           'keypressfcn', callstr, ...
                                           'windowstyle', 'normal', ...
                                           'windowstate', 'maximized',...
                                           'numbertitle', 'off', ...
                                           'position', [500 500 500 500], ...
                                           'userdata', 'timeout');
            throttle = throttle_prev;
            steer =steer_prev;
            brake = brake_prev;
            reverse_gear = reverse_gear_prev;
            manualcontrol = manualcontrol_prev;
            autopilot = autopilot_prev;
            
            throttle = single(throttle);
            steer = single(steer);
            reverse_gear = boolean(reverse_gear);
            brake = single(brake);
            manualcontrol = boolean(manualcontrol);
            autopilot = boolean(autopilot);
                                       
            
        end

        function [steer,throttle,reverse_gear,brake,manualcontrol,autopilot] = stepImpl(obj,steer_prev,throttle_prev,reverse_gear_prev,brake_prev,manualcontrol_prev,autopilot_prev)
            
            
            
            
            % Keys function
            % 
            %   W       --->    Forward
            %   S       --->    Brake
            %   A       --->    Left
            %   D       --->    Right
            %   R       --->    Reverse gear
            %   F       --->    Forward gear
            %   E       --->    Enter auto control
            %   Q       --->    Quit auto control

            obj.key = lower(obj.key);
            
            throttle = throttle_prev;
            steer = steer_prev;
            brake = brake_prev;
            reverse_gear = reverse_gear_prev;
            manualcontrol = manualcontrol_prev;
            autopilot = autopilot_prev;
            
            throttle = single(throttle);
            steer = single(steer);
            reverse_gear = boolean(reverse_gear);
            brake = single(brake);
            manualcontrol = boolean(manualcontrol);
            autopilot = boolean(autopilot);
            % Handle keyboard
            
            dt = cputime - obj.time_last_update;
            
            try
                obj.key = get(obj.keyboard_handle,'Userdata');
                if strcmp(obj.key, 'timeout')
                    return;
                end

                % Change the drive mode
                if strcmp(obj.key, 'e')
                    autopilot = true;
                    manualcontrol = false;
                    manualcontrol = boolean(manualcontrol);
                    autopilot = boolean(autopilot);
                    
                elseif strcmp(obj.key, 'q')
                    autopilot = false;
                    manualcontrol = true;
                    
                % Throttle
                elseif strcmp(obj.key, 'w')
                    throttle = throttle + 0.2 * dt;
                    brake = 0;
                    steer = 0;
                    throttle = single(throttle);
                    brake = single(brake);
                    steer = single(steer);
                    
                    
                    if throttle > 0.6
                        throttle = 0.6;
                        throttle = single(throttle);
                    end


                elseif strcmp(obj.key, 's')
                    throttle = 0;
                    brake = 1;
                    throttle = single(throttle);
                    brake = single(brake);
     
                % Steer
                elseif strcmp(obj.key, 'd')

                    steer = steer + 0.2 * dt;
                    steer = single(steer);

                    if steer > 0.6
                        steer = 0.6;
                        steer = single(steer);
                    end

                elseif strcmp(obj.key, 'a')

                    steer = steer - 0.2 * dt;
                    steer = single(steer);

                    if steer < -0.6
                        steer = -0.6;
                        steer = single(steer);
                    end

                % Direction
                elseif strcmp(obj.key, 'r')
                    
                    reverse_gear = true;
                    reverse_gear = boolean(reverse_gear);

                elseif strcmp(obj.key, 'f')

                    reverse_gear = false;
                    reverse_gear = boolean(reverse_gear);
       
                end
                
            catch
                    autopilot = true;
                    autopilot = boolean(autopilot);
            end

            % Update time
            obj.time_last_update = cputime;
            
            
        end
        
        function [steer,throttle,reverse_gear,brake,manualcontrol,autopilot] = isOutputComplexImpl(~)
            throttle = false;
            steer = false;
            reverse_gear = false;
            brake = false;
            manualcontrol = false;
            autopilot = false;
        end
        
        function [steer,throttle,reverse_gear,brake,manualcontrol,autopilot] = getOutputSizeImpl(obj)
            throttle = 1;
            steer = 1;
            reverse_gear = 1;
            brake = 1;
            manualcontrol = 1;
            autopilot = 1;
        end
        
        function [steer,throttle,reverse_gear,brake,manualcontrol,autopilot] = getOutputDataTypeImpl(~)
            throttle = 'single';
            steer = 'single';
            reverse_gear = 'boolean';
            brake = 'single';
            manualcontrol = 'boolean';
            autopilot = 'boolean';
        end

        function [steer,throttle,reverse_gear,brake,manualcontrol,autopilot] = isOutputFixedSizeImpl(~)
            throttle = true;
            steer = true;
            reverse_gear = true';
            brake = true;
            manualcontrol = true;
            autopilot = true;
        end
        
        function resetImpl(~)
            % Initialize / reset discrete-state properties
        end
    end
    
    methods(Access= public)
        function delete(obj)
            % Close the figure if it still exists
            if ~isempty(obj.keyboard_handle)
                close(obj.keyboard_handle);
            end
            
        end
    end
end
