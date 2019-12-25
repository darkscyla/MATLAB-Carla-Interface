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
        % Semantic segmentation resolution
        width = 960;
        height = 540;
        
        % RGB resolution
        rgb_width = 1920;
        rgb_height = 1080;
        
        tesla;
        last_position;
        
        ss_rgb;
        module_ss_rgb;
        
        rgb;
        module_rgb;
        
        last_update_time;
        history_array = zeros(2, 8);
        steer_spline;
        
        steer_window_avg = 0;
        push_back_time = 0;
        frames_since_last_update = 0;
        
        bias = -0.033;
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
            spawn_point.location.x = -88.8;
            spawn_point.location.y = 149.0;
            spawn_point.rotation.yaw = 90;
            
            obj.tesla = world.spawn_actor(car_bp, spawn_point);
            obj.tesla.set_autopilot(false);
            
            control = obj.tesla.get_control();
            control.throttle = 0.6;
            control.steer = 0;
            obj.tesla.apply_control(control);
            
            % Semantic Segmentation
            blueprint = world.get_blueprint_library().find('sensor.camera.semantic_segmentation');
            blueprint.set_attribute('image_size_x', num2str(obj.width))
            blueprint.set_attribute('image_size_y', num2str(obj.height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
            obj.ss_rgb = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_ss_rgb = sensorBind(obj.ss_rgb, "ss_rgb", "semantic_segmentation_rgb", "array");
            
            
            % RGB
            blueprint = world.get_blueprint_library().find('sensor.camera.rgb');
            blueprint.set_attribute('image_size_x', num2str(obj.rgb_width))
            blueprint.set_attribute('image_size_y', num2str(obj.rgb_height))
            
            transform = py.carla.Transform(py.carla.Location(pyargs('x',-5.0, 'z',2.5)));
            obj.rgb = world.spawn_actor(blueprint, transform, pyargs('attach_to',obj.tesla));

            obj.module_rgb = sensorBind(obj.rgb, "rgb", "rgb", "array");
            
            obj.last_position = obj.tesla.get_location();
            
            obj.last_update_time = cputime;
        end

        function [SEMANTIC_SEGMENTATION_RGB, RGB] = stepImpl(obj)
            
            % Semantic Segmentation 
            SEMANTIC_SEGMENTATION_RGB = uint8(py.getattr(obj.module_ss_rgb, 'array'));
            
            %% Lane Detection
            control = obj.tesla.get_control();

            % Crop the image to region of interest
            horizontal_crop = 0.0;
            hor_start_pos = ceil(obj.width/2);
            hor_end_pos = floor(obj.width * (1 - horizontal_crop / 2));
            
            vertical_crop = .5;
            vert_start_pos = 1 + ceil(obj.height * vertical_crop);
            
            filtered_lanes = SEMANTIC_SEGMENTATION_RGB(vert_start_pos:obj.height,hor_start_pos:hor_end_pos,1) == 157;

            
            % Apply hough transformations to get the lines from the sensor
            % image
            BW = filtered_lanes;
            [H,T,R] = hough(BW);
            P  = houghpeaks(H,5,'threshold',ceil(0.0*max(H(:))));
            lines = houghlines(BW,T,R,P,'FillGap',50,'MinLength',20);
            
            % Calculate the angles of the lines based on the x-y
            % coordinates
            angles = zeros(1, length(lines));
            for k = 1:length(lines)
                angles(k) = rad2deg(atan((lines(k).point1(2) - lines(k).point2(2))/(lines(k).point1(1) - lines(k).point2(1))));
            end
            
            % Normally from the camera position, the angle between the road
            % and the car if the car is moving straight
            ref_angle = 57;
            max_angle = max(angles);
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Loop to draw the nearest right lanes detected
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            imshow(filtered_lanes), hold on

            for k = 1:length(lines) 
               angle = rad2deg(atan((lines(k).point1(2) - lines(k).point2(2))/(lines(k).point1(1) - lines(k).point2(1))));
               if ~(angle == max_angle)
                  continue
               end

               xy = [lines(k).point1; lines(k).point2];
               plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

               % Plot beginnings and ends of lines
               plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
               plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

            end

            set(gcf, 'Visible', 'on');
            hold off;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            % If the vehicle has just spawned and not started moving, dont
            % update steer history 
            started = false;
            current_location = obj.tesla.get_location();
            thres_loc = 0.001;
            
            if abs(current_location.x - obj.last_position.x) > thres_loc...
                    && abs(current_location.y - obj.last_position.y) > thres_loc
                started = true;
            end
            
            % Calculate the time it took to process one frame. Really
            % necessary to make the calculations independent of the frame
            % rate. Otherwise, running simulation on a pc that can run it
            % on 20 FPS will steer 20 times vs 2 FPS steers 2 times, dt 
            % scales that
            dt = cputime - obj.last_update_time;
            fac = 0.5 * tanh(0.1 * abs(ref_angle - max_angle)) * dt;
            
            obj.last_update_time = cputime;
            
            
            % How much effect the last steering has on the future steer
            mem = 0.0025;
            
            % Filter out lines that are almost horizontal
            if ~isempty(fac) && ((max_angle < 85) && (max_angle > 25)) && started
                if max_angle < ref_angle                    
                    steer = mem * control.steer + fac;
                else
                    steer = mem * control.steer - fac;
                end
            else
                steer = obj.bias;
            end
            
            % Update the spline history ~ every .2 seconds. A very high FPS
            % would polute the spline fitting
            if cputime - obj.push_back_time > .2
                
                % Delete the oldest element in the array
                obj.history_array(:,1) = [];
                
                % Most recent history weight
                weight = 1.25;
                
                % Average of the steer since the last update
                steer = (obj.steer_window_avg + steer)/(obj.frames_since_last_update + 1);

                % Add to the histroy
                obj.history_array = [obj.history_array, [cputime;steer]];

                % Pre-processing for the curve fitting
                [timeData, steerData] = prepareCurveData( obj.history_array(1,:),...
                                                          obj.history_array(2,:));


                obj.history_array(end) = obj.history_array(end) * weight;

                % Fit a smoothing spline to the data
                obj.steer_spline = fit( timeData, steerData, 'smoothingspline', ...
                                        'Normalize', 'on', 'SmoothingParam',0.95);

                obj.history_array(end) = obj.history_array(end) / weight;
                
                obj.steer_window_avg = 0;
                obj.frames_since_last_update = 0;
                
                obj.push_back_time = cputime;
                
            else
                % If not yet time to update, keep track of the steer values
                % in this time slot
                obj.steer_window_avg = obj.steer_window_avg + steer;
                obj.frames_since_last_update = obj.frames_since_last_update + 1;
            end
            
            % Predict and steer the car
            control.steer = obj.steer_spline(cputime + dt/2);
                                                  
            obj.tesla.apply_control(control);            
            obj.last_position = current_location;
            
            %% RGB sensor
            RGB = uint8(py.getattr(obj.module_rgb, 'array'));
        end
        
        function [SEMANTIC_SEGMENTATION_RGB, RGB] = isOutputComplexImpl(~)
            SEMANTIC_SEGMENTATION_RGB = false;
            RGB = false;
        end
        
        function [SEMANTIC_SEGMENTATION_RGB, RGB] = getOutputSizeImpl(obj)
            SEMANTIC_SEGMENTATION_RGB = [obj.height obj.width 3];
            RGB = [obj.rgb_height obj.rgb_width 3];
        end
        
        function [SEMANTIC_SEGMENTATION_RGB, RGB] = getOutputDataTypeImpl(~)
            SEMANTIC_SEGMENTATION_RGB = 'uint8';
            RGB = 'uint8';
        end

        function [SEMANTIC_SEGMENTATION_RGB, RGB] = isOutputFixedSizeImpl(~)
            SEMANTIC_SEGMENTATION_RGB = true;
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
            
            % Semantic segmentation sensor
            if ~isempty(obj.ss_rgb)
                obj.ss_rgb.destroy();
            end

            % RGB camera
            if ~isempty(obj.rgb)
                obj.rgb.destroy();
            end
        end
    end
end
    