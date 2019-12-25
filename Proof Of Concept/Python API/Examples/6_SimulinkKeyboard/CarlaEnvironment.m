classdef CarlaEnvironment < matlab.System & matlab.system.mixin.Propagates
    % Untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties
        % Other actors
        npc_to_spawn = 20;
    end

    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)
        % Sensors resolution
        width = 1920;
        height = 1080;
        
        tesla;
        npc_list;
        
        keyboard_handle;

        rgb;
        module_rgb;
        time_last_update = 0;
    end

    methods(Access = protected)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            port = int16(2000);
            client = py.carla.Client('localhost', port);
            client.set_timeout(2.0);
            world = client.get_world();
            
            % Spawn Vehicle
            blueprint_library = world.get_blueprint_library();
            car_list = py.list(blueprint_library.filter("model3"));
            car_bp = car_list{1};
            spawn_point = py.random.choice(world.get_map().get_spawn_points());
            obj.tesla = world.spawn_actor(car_bp, spawn_point);
            obj.tesla.set_autopilot(true);
            
            % Spawn NPC's
            npc_bps = blueprint_library.filter("vehicle");
            
            i = 1;
            while i <= obj.npc_to_spawn
                try
                    npc_bp = py.random.choice(npc_bps);
                    spawn_point = py.random.choice(world.get_map().get_spawn_points());
                    obj.npc_list{i} = world.spawn_actor(npc_bp, spawn_point);
                    obj.npc_list{i}.set_autopilot(true);
                catch
                    % In case spawing fails due to collision, try again
                    i = i - 1;
                end
                i = i + 1;
            end
         
            % RGB
            blueprint = world.get_blueprint_library().find('sensor.camera.rgb');
            blueprint.set_attribute('image_size_x', num2str(obj.width))
            blueprint.set_attribute('image_size_y', num2str(obj.height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
            obj.rgb = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_rgb = sensorBind(obj.rgb, "rgb", "rgb", "array");
            
            % Handle keyboard
            callstr = 'set(gcbf, ''Userdata'', get(gcbf, ''Currentkey'')) ; uiresume ' ;
            obj.keyboard_handle = figure( 'name', 'Press a key', ...
                                           'keypressfcn', callstr, ...
                                           'windowstyle', 'modal', ...
                                           'numbertitle', 'off', ...
                                           'position', [0 0 1 1], ...
                                           'userdata', 'timeout');
            set(obj.keyboard_handle,'Visible','on');
        end

        function [RGB] = stepImpl(obj)
            % RGB
            RGB = uint8(py.getattr(obj.module_rgb, 'array'));
            
            % Handle keyboard
            dt = cputime - obj.time_last_update;
            
            try
                keyPressed = get(obj.keyboard_handle,'Userdata');
                processKeyboard(obj.tesla, keyPressed, dt);
                
                % Reset the pressed key
                set(obj.keyboard_handle,'Userdata', char(0));
            catch
                obj.tesla.set_autopilot(true);
            end
            
            % Update time
            obj.time_last_update = obj.time_last_update + dt;
            pause(0.025);
        end
        
        function [RGB] = isOutputComplexImpl(~)
            RGB = false;
        end
        
        function [RGB] = getOutputSizeImpl(obj)
            RGB = [obj.height obj.width 3];
        end
        
        function [RGB] = getOutputDataTypeImpl(~)
            RGB = 'uint8';
        end

        function [RGB] = isOutputFixedSizeImpl(~)
            RGB = true;
        end
        
        function resetImpl(~)
            % Initialize / reset discrete-state properties
        end
    end
    
    methods(Access= public)
        function delete(obj)
            % Delete the car from the Carla world
            if ~isempty(obj.tesla)
                obj.tesla.destroy();
            end
            
            % Delete NPCs
            if ~isempty(obj.npc_list)
                for i=1:obj.npc_to_spawn
                    obj.npc_list{i}.destroy();
                end
            end
            
            % RGB camera
            if ~isempty(obj.rgb)
                obj.rgb.destroy();
            end
            
            % Close the figure if it still exists
            if ~isempty(obj.keyboard_handle)
                close(obj.keyboard_handle);
            end
            
        end
    end
end
    