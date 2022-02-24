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
        % Sensors resolution
        width = 960;
        height = 540;
        
        tesla
        
        ss_rgb;
        module_ss_rgb;
        
        rgb;
        module_rgb;
                
        depthRGB;
        module_depthRGB;
        
        depth;
        module_depth;
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
            obj.tesla = world.spawn_actor(car_bp, spawn_point);
            obj.tesla.set_autopilot(true);
            
            % Semantic Segmentation
            blueprint = world.get_blueprint_library().find('sensor.camera.semantic_segmentation');
            blueprint.set_attribute('image_size_x', num2str(obj.width))
            blueprint.set_attribute('image_size_y', num2str(obj.height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
            obj.ss_rgb = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_ss_rgb = sensorBind(obj.ss_rgb, "ss_rgb", "semantic_segmentation_rgb", "array");
            
            
            % RGB
            blueprint = world.get_blueprint_library().find('sensor.camera.rgb');
            blueprint.set_attribute('image_size_x', num2str(obj.width))
            blueprint.set_attribute('image_size_y', num2str(obj.height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
            obj.rgb = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_rgb = sensorBind(obj.rgb, "rgb", "rgb", "array");
            
            
            % Depth RGB
            blueprint = world.get_blueprint_library().find('sensor.camera.depth');
            blueprint.set_attribute('image_size_x', num2str(obj.width))
            blueprint.set_attribute('image_size_y', num2str(obj.height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
            obj.depthRGB = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_depthRGB = sensorBind(obj.depthRGB, "depthRGB", "depthRGB", "array");
            

            % Depth (intensity map)
            blueprint = world.get_blueprint_library().find('sensor.camera.depth');
            blueprint.set_attribute('image_size_x', num2str(obj.width))
            blueprint.set_attribute('image_size_y', num2str(obj.height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
            obj.depth = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_depth = sensorBind(obj.depth, "depth", "depth", "array");
        end

        function [SEMANTIC_SEGMENTATION_RGB, RGB, DEPTH_RBG, DEPTH] = stepImpl(obj)
            
            % Semantic Segmentation 
            SEMANTIC_SEGMENTATION_RGB = uint8(py.getattr(obj.module_ss_rgb, 'array'));

            % RGB
            RGB = uint8(py.getattr(obj.module_rgb, 'array'));
            
            % Depth RGB
            DEPTH_RBG = uint8(py.getattr(obj.module_depthRGB, 'array'));
            
            % Depth intensity map
            DEPTH = double(py.getattr(obj.module_depth, 'array'));
        end
        
        function [SEMANTIC_SEGMENTATION_RGB, RGB, DEPTH_RBG, DEPTH] = isOutputComplexImpl(~)
            SEMANTIC_SEGMENTATION_RGB = false;
            RGB = false;
            DEPTH_RBG = false;
            DEPTH = false;
        end
        
        function [SEMANTIC_SEGMENTATION_RGB, RGB, DEPTH_RBG, DEPTH] = getOutputSizeImpl(obj)
            SEMANTIC_SEGMENTATION_RGB = [obj.height obj.width 3];
            RGB = [obj.height obj.width 3];
            DEPTH_RBG = [obj.height obj.width 3];
            DEPTH = [obj.height obj.width];
        end
        
        function [SEMANTIC_SEGMENTATION_RGB, RGB, DEPTH_RBG, DEPTH] = getOutputDataTypeImpl(~)
            SEMANTIC_SEGMENTATION_RGB = 'uint8';
            RGB = 'uint8';
            DEPTH_RBG = 'uint8';
            DEPTH = 'double';
        end

        function [SEMANTIC_SEGMENTATION_RGB, RGB, DEPTH_RBG, DEPTH] = isOutputFixedSizeImpl(~)
            SEMANTIC_SEGMENTATION_RGB = true;
            RGB = true;
            DEPTH_RBG = true;
            DEPTH = true;
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
            
            % Semantic segmentation sensor
            if ~isempty(obj.ss_rgb)
                obj.ss_rgb.destroy();
            end

            % RGB camera
            if ~isempty(obj.rgb)
                obj.rgb.destroy();
            end
            
            % Depth camera rgb
            if ~isempty(obj.depthRGB)
                obj.depthRGB.destroy();
            end
            
            % Depth camera
            if ~isempty(obj.depth)
                obj.depth.destroy();
            end
        end
    end
end
