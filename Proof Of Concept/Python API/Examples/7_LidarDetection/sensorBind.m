function pyModule = sensorBind(sensor, fileName, sensorType, varName)

    %% Function signature
    % ---------------------------------------------------------------------
    %                               Inputs
    % ---------------------------------------------------------------------
    % sensor        ---> The sensor 
    % fileName      ---> Name of the python file that will be produced for
    %                    sensor binding
    %
    % varName       ---> Name of the variable that will store the sensor data
    % 
    % sensorType    ---> Choose from sensors
    %                 1. rgb                     	---> Normal camera
    %                 2. grayScale               	---> grayScale image
    %                 3. depth                   	---> Gives distances in m
    %				  4. depthRGB				 	---> RGB coded distance
    %                 5. semantic_segmentation   	---> Classifies objects
    %                                                 	 and tags them with id
    %				  6. semantic_segmentation_rgb	---> Map the id to RGB for 
    %   												 visualization 
    %                 7. lidar                		---> Gives 3d points array 
    %                                                    along with intensity values
    %
    % ---------------------------------------------------------------------
    %                               Output
    % ---------------------------------------------------------------------
    % pyModule ---> Returns a python module that is responsible for
    %               acquring the sensor data 
    
    %% Check if the sensor is valid
    if ~isa(sensor, 'py.carla.libcarla.ServerSideSensor')
        error("The provided sensor is not valid\n")
    end
    
    %% Create sensor callback binder
    if strcmp(sensorType, "rgb") || strcmp(sensorType, "grayScale") || strcmp(sensorType, "depthRGB")
        rgb(fileName, sensorType, varName);
    elseif strcmp(sensorType, "depth")
        depth(fileName, sensorType, varName);
    elseif strcmp(sensorType, "semantic_segmentation")
        semantic_segmentation(fileName, sensorType, varName);
    elseif strcmp(sensorType, "semantic_segmentation_rgb")
        semantic_segmentation_rgb(fileName, sensorType, varName);
    elseif strcmp(sensorType, "lidar")
        lidar(fileName, sensorType, varName);
    end
    
    pyModule = py.importlib.import_module(fileName);
    eval(strcat("py.", fileName, ".bindSensor(sensor)"));
    
    % Takes time for the sensor to recieve first data
    pause(0.25);
end