port = int16(2000);
client = py.carla.Client('localhost', port);
client.set_timeout(10.0);
world = client.get_world();
blueprint_library = world.get_blueprint_library();
actor_list = py.list(world.get_actors());

id = 0;

for i=1:length(actor_list)
    k = actor_list{i}.type_id;
    
    if k(1:7) == "vehicle"
        id = i;
    end
end

if (id == 0)
    fprintf('ERROR: Could not find any vehicle!\n');
    fprintf('The program will now exit!\n');
    return;
end

carla_is_running = true;

blueprint = world.get_blueprint_library().find('sensor.camera.rgb');
blueprint.set_attribute('image_size_x', '1920')
blueprint.set_attribute('image_size_y', '1080')
% blueprint.set_attribute('sensor_tick', '.1');
transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
sensor = world.spawn_actor(blueprint, transform, pyargs('attach_to',actor_list{id}));

pyModule = py.importlib.import_module('senFunc');
py.senFunc.getData(sensor);

pause(0.5);

% Load YOLO model
if exist('yoloml','var') ~= 1
    disp('loading modified network')
    load('yoloml.mat')
end

% Display Initial Frame
image = py.getattr(pyModule, 'array');
imMat = uint8(image);

% Set Yolo parameters
probThresh = 0.175;
iouThresh = 0.4;

myImageSize = size(imMat);

classLabels = ["aeroplane",	"bicycle",	"bird"	,"boat",	"bottle"	,"bus"	,"car",...
"cat",	"chair"	,"cow"	,"diningtable"	,"dog"	,"horse",	"motorbike",	"person",	"pottedplant",...
"sheep",	"sofa",	"train",	"tvmonitor"];

imageHandle = imshow(imMat);

% Call "Detect" After how many frames
totalFramesElapsed = 0;
predictEvery = 1;   

while carla_is_running
    
    image = py.getattr(pyModule, 'array');
    imMat = uint8(image);
    
    redImage = single(imresize(imMat,[448 448]))/255;

    if ~rem(totalFramesElapsed, predictEvery)
        out = predict(yoloml,redImage,'ExecutionEnvironment','gpu');
    end
    
    class = out(1:980);
    boxProbs = out(981:1078);
    boxDims = out(1079:1470);

    outArray = zeros(7,7,30);

    for j = 0:6
        for i = 0:6
            outArray(i+1,j+1,1:20) = class(i*20*7+j*20+1:i*20*7+j*20+20);
            outArray(i+1,j+1,21:22) = boxProbs(i*2*7+j*2+1:i*2*7+j*2+2);
            outArray(i+1,j+1,23:30) = boxDims(i*8*7+j*8+1:i*8*7+j*8+8);
        end
    end
    
    [cellProb, cellIndex] = max(outArray(:,:,21:22),[],3);
    contain = max(outArray(:,:,21:22),[],3)>probThresh;

    [classMax, classMaxIndex] = max(outArray(:,:,1:20),[],3);

    boxes = [];

    counter = 0;
    
    for i = 1:7
        for j = 1:7
            if contain(i,j) == 1
                counter = counter+1;           

                % Bounding box center relative to cell
                x = outArray(i,j,22+1+(cellIndex(i,j)-1)*4);
                y = outArray(i,j,22+2+(cellIndex(i,j)-1)*4);

                % Yolo outputs the square root of the width and height of the
                % bounding boxes (subtle detail in paper that took me forver to realize). 
                % Relative to image size.
                w = (outArray(i,j,22+3+(cellIndex(i,j)-1)*4))^2;
                h = (outArray(i,j,22+4+(cellIndex(i,j)-1)*4))^2;

                %absolute values scaled to image size
                wS = w*448; 
                hS = h*448;
                xS = (j-1)*448/7+x*448/7-wS/2;
                yS = (i-1)*448/7+y*448/7-hS/2;

                % this array will be used for drawing bounding boxes in Matlab
                boxes(counter).coords = [xS yS wS hS]; 

                % save cell indices in the structure
                boxes(counter).cellIndex = [i,j];

                % save classIndex to structure
                boxes(counter).classIndex = classMaxIndex(i,j);    

                % save cell proability to structure
                boxes(counter).cellProb = cellProb(i,j);

                % put in a switch for non max which we will use later
                boxes(counter).nonMax = 1;
            end            
        end
    end

    for i = 1:length(boxes)
        for j = i+1:length(boxes)
            % calculate intersection over union (can also use bboxOverlapRatio
            % with proper toolbox
            intersect = rectint(boxes(i).coords,boxes(j).coords);
            union = boxes(i).coords(3)*boxes(i).coords(4)+boxes(j).coords(3)*boxes(j).coords(4)-intersect;
            iou(i,j) = intersect/union;

            if boxes(i).classIndex == boxes(j).classIndex && iou(i,j) > iouThresh                
                [value(i), dropIndex(i)] = min([boxes(i).cellProb boxes(j).cellProb]);
                if dropIndex(i) == 1
                    boxes(i).nonMax=0;
                elseif dropIndex(i) == 2
                    boxes(j).nonMax=0;                
                end
            end                
        end
    end 
    
    for i = 1:length(boxes)
        if boxes(i).nonMax == 1
            textStr = convertStringsToChars(classLabels(boxes(i).classIndex));
             
            if sum(textStr == [ "bicycle", "car", "motorbike",	"person"])
                position = [(boxes(i).cellIndex(2)-1)*myImageSize(2)/7 (boxes(i).cellIndex(1)-1)*myImageSize(1)/7];
                transformedCoor = boxes(i).coords .* [myImageSize(2) myImageSize(1) myImageSize(2) myImageSize(1)] / 448;
                
                imMat = insertText(imMat, transformedCoor([1, 2]), textStr, 'AnchorPoint','LeftBottom', 'FontSize', 36);
                imMat = insertShape(imMat, 'Rectangle', transformedCoor, 'LineWidth',5);
                
            end
        end
    end
     
    set(imageHandle,'Cdata',imMat);
    drawnow;

    totalFramesElapsed = totalFramesElapsed + 1;
     
    pause(0.001);
      
end