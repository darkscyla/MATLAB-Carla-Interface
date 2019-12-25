function rgb(fileName, sensorType, varName)
    
    %% Check if input is valid
    if ~isstring(fileName) && ~ischar(fileName)
        error("File name must a string or char array\n")
    end
    
    if ~strcmp(sensorType, "rgb") && ~strcmp(sensorType,"grayScale") && ~strcmp(sensorType, "depthRGB")
       error("Wrong sensor selected\n") 
    end

    if ~isvarname(varName)
        error("Invalid variable name\n")
    end
    
    %% Create a python file 
    file = fopen(strcat(fileName, '.py'), 'w');
    
    % Automatically genrates a python file containing the sensor call back
    % bindings
    fprintf(file, 'import numpy as np\n');
    fprintf(file, '\n');
    fprintf(file, 'def bindSensor(sensor):\n');
    fprintf(file, '    sensor.listen(lambda _image: do_something(_image))\n');
    fprintf(file, '\n');
    fprintf(file, 'def do_something(_image):\n');
    fprintf(file, '    global %s\n', varName);
    fprintf(file, '    data = np.frombuffer(_image.raw_data, dtype=np.dtype("uint8"))\n');
    fprintf(file, '\n');
    fprintf(file, '    # Convert to RGB from BGRA\n');
    fprintf(file, '    data = np.reshape(data, (_image.height, _image.width, 4))\n');
    fprintf(file, '    data = data[:, :, :3]\n');
    fprintf(file, '    data = data[:, :, ::-1]\n');
    fprintf(file, '\n');
    
    if sensorType == "grayScale"
       fprintf(file, '    # Convert it to gray-scale\n');
       fprintf(file, '    data = np.dot(data, [0.2989, 0.5870, 0.1140])\n');
       fprintf(file, '\n');
    end
    
    fprintf(file, '    # Convert the data into MATLAB cast compatible type\n');
    fprintf(file, '    %s = np.ascontiguousarray(data)\n', varName);
    
    fclose(file);

end