function semantic_segmentation(fileName, sensorType, varName)

    %% Check if input is valid
    if ~isstring(fileName) && ~ischar(fileName)
        error("File name must a string or char array\n")
    end
    
    if ~strcmp(sensorType, "semantic_segmentation")
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
    fprintf(file, '    # Get the red channel which has the tags\n');
    fprintf(file, '    data = np.reshape(data, (_image.height, _image.width, 4))\n');
    fprintf(file, '    data = data[:, :, 2]\n');
    fprintf(file, '\n');
    
    fprintf(file, '    # Convert the data into MATLAB cast compatible type\n');
    fprintf(file, '    %s = np.ascontiguousarray(data)\n', varName);
    
    fclose(file);

end