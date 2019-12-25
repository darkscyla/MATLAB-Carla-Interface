classdef CarlaEnvironment < matlab.System & matlab.system.mixin.Propagates
    % Untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties
        
    end

    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)
        car;
    end

    methods(Access = protected)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            port = int16(2000);
            client = py.carla.Client('localhost', port);
            client.set_timeout(10.0);
            world = client.get_world();
            
            % Spawn Vehicle
            blueprint_library = world.get_blueprint_library();
            car_list = py.list(blueprint_library.filter("model3"));
            car_bp = car_list{1};
            spawn_point = py.random.choice(world.get_map().get_spawn_points());
            obj.car = world.spawn_actor(car_bp, spawn_point);
            obj.car.set_autopilot(true);

        end

        function [x_position, x_velocity] = stepImpl(obj)
            % 
            pause(0.001);
            x_position = obj.car.get_location().x;
            x_velocity = obj.car.get_velocity().x;
        end
        
        function [distance, velocity] = isOutputComplexImpl(~)
            distance = false;
            velocity = false;
        end
        
        function [distance, velocity] = getOutputSizeImpl(~)
            distance = [1 1];
            velocity = [1 1];
        end
        
        function [distance, velocity] = getOutputDataTypeImpl(~)
            distance = 'double';
            velocity = 'double';
        end

        function [distance, velocity] = isOutputFixedSizeImpl(~)
            distance = true;
            velocity = true;
        end
        
        function resetImpl(~)
            % Initialize / reset discrete-state properties
        end
    end
    
    methods(Access= public)
        function delete(obj)
            % Delete the car from the Carla world
            if ~isempty(obj.car)
                obj.car.destroy();
            end
        end
    end
end
