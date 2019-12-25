classdef LaneDetection < matlab.System & matlab.system.mixin.Propagates
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
        width;
        height;

        last_position;
        ss_rgb;
        module_ss_rgb;
        
        last_update_time;
        history_array;
        steer_spline;
        
        steer_window_avg;
        push_back_time;
        frames_since_last_update;
        
        bias;
    end

    methods(Access = protected)
        function [SEMANTIC_SEGMENTATION,steer,throttle,brake,manualcontrol,autopilot] = setupImpl(obj,steer_prev,throttle_prev,brake_prev,manualcontrol_prev,autopilot_prev,SEMANTIC_IN)
            % Perform one-time calculations, such as computing constants
                               
            throttle = throttle_prev;
            steer = steer_prev;
            brake = brake_prev;
            manualcontrol = manualcontrol_prev;
            autopilot = autopilot_prev;
            
            % Semantic Segmentation 
            SEMANTIC_SEGMENTATION = uint8(SEMANTIC_IN);

            obj.history_array = zeros(2, 8);
            % Semantic segmentation resolution
            obj.width = 960;
            obj.height = 540;
           
            obj.steer_window_avg = 0;
            obj.push_back_time = 0;
            obj.frames_since_last_update = 0;

            obj.bias = -0.033;
            obj.last_update_time = cputime;
            
        end

        function [SEMANTIC_SEGMENTATION,steer,throttle,brake,manualcontrol,autopilot] = stepImpl(obj,steer_prev,throttle_prev,brake_prev,manualcontrol_prev,autopilot_prev,SEMANTIC_IN)
            
            % Semantic Segmentation 
            SEMANTIC_SEGMENTATION = uint8(SEMANTIC_IN);
            
            %% Lane Detection
            throttle = 0.4; 
            steer = steer_prev;
            brake = brake_prev;
            manualcontrol = manualcontrol_prev;
            autopilot = autopilot_prev;
            
            throttle = single(throttle);
            steer = single(steer);
            brake = single(brake);
            manualcontrol = boolean(manualcontrol);
            autopilot = boolean(autopilot);

            % Crop the image to region of interest
            horizontal_crop = 0.0;
            hor_start_pos = ceil(obj.width/2);
            hor_end_pos = floor(obj.width * (1 - horizontal_crop / 2));
            
            vertical_crop = .5;
            vert_start_pos = 1 + ceil(obj.height * vertical_crop);
            
            filtered_lanes = SEMANTIC_SEGMENTATION(vert_start_pos:obj.height,hor_start_pos:hor_end_pos,1) == 157;

            
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

            hold off;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            
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
            if ~isempty(fac) && ((max_angle < 85) && (max_angle > 25))
                if max_angle < ref_angle                    
                    steer = mem * steer + fac;
                else
                    steer = mem * steer - fac;
                end
            else
                steer = obj.bias;
            end
            
            % Update the spline history ~ every .2 seconds. A very high FPS
            % would polute the spline fitting
            if cputime - obj.push_back_time > .2
                
                %Delete the oldest element in the array
                obj.history_array(:,1) = [];
                
                %Most recent history weight
                weight = 1.25;
                
                %Average of the steer since the last update
                steer = (obj.steer_window_avg + steer)/(obj.frames_since_last_update + 1);

                %Add to the histroy
                obj.history_array = [obj.history_array, [cputime;steer]];

                %Pre-processing for the curve fitting
                [timeData, steerData] = prepareCurveData( obj.history_array(1,:),...
                                                          obj.history_array(2,:));


                obj.history_array(end) = obj.history_array(end) * weight;

                %Fit a smoothing spline to the data
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
            steer = obj.steer_spline(cputime + dt/2);
            
            throttle = single(throttle);
            steer = single(steer);
            brake = single(brake);
            manualcontrol = boolean(manualcontrol);
            autopilot = boolean(autopilot);
                                                           
            
        end
        
        function [SEMANTIC_SEGMENTATION,throttle,steer,brake,manualcontrol,autopilot] = isOutputComplexImpl(~)
            SEMANTIC_SEGMENTATION = false;
            throttle = false;
            steer = false;
            brake = false;
            manualcontrol = false;
            autopilot = false;
        end
        
        function [SEMANTIC_SEGMENTATION,throttle,steer,brake,manualcontrol,autopilot] = getOutputSizeImpl(obj)
            SEMANTIC_SEGMENTATION = [540 960 3];
            throttle = 1;
            steer = 1;
            brake = 1;
            manualcontrol = 1;
            autopilot = 1;
        end
        
        function [SEMANTIC_SEGMENTATION,throttle,steer,brake,manualcontrol,autopilot] = getOutputDataTypeImpl(~)
            SEMANTIC_SEGMENTATION = 'uint8';
            throttle = 'single';
            steer = 'single';
            brake = 'single';
            manualcontrol = 'boolean';
            autopilot = 'boolean';
        end

        function [SEMANTIC_SEGMENTATION,throttle,steer,brake,manualcontrol,autopilot] = isOutputFixedSizeImpl(~)
            SEMANTIC_SEGMENTATION = true;
            throttle = true;
            steer = true;
            brake = true;
            manualcontrol = true;
            autopilot = true;
        end
        
        function resetImpl(~)
            % Initialize / reset discrete-state properties
        end
    end
end
    
