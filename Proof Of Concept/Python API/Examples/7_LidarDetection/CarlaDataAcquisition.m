port = int16(2000);
client = py.carla.Client('localhost', port);
client.set_timeout(10.0);
world = client.get_world();

% Spawn Vehicle
blueprint_library = world.get_blueprint_library();
car_list = py.list(blueprint_library.filter("model3"));
car_bp = car_list{1};
spawn_point = py.random.choice(world.get_map().get_spawn_points());
tesla = world.spawn_actor(car_bp, spawn_point);
tesla.set_autopilot(true);

carla_is_running = true;

% Sensor 1
blueprint = world.get_blueprint_library().find('sensor.camera.rgb');
blueprint.set_attribute('image_size_x', '960')
blueprint.set_attribute('image_size_y', '540')
% blueprint.set_attribute('sensor_tick', '.1');
transform = py.carla.Transform(py.carla.Location(pyargs('x',-7.5, 'z',2.5)));
sensor = world.spawn_actor(blueprint, transform, pyargs('attach_to',tesla));

pyModule = sensorBind(sensor, "rgb", "rgb", "array");
currentImage = uint8(py.getattr(pyModule, 'array'));
imageHandle = imshow(currentImage);
set(gca,'units','pixels'); % set the axes units to pixels
x = get(gca,'position'); % get the position of the axes
set(gcf,'units','pixels'); % set the figure units to pixels
y = get(gcf,'position'); % get the figure position
set(gcf,'position',[y(1) y(2) x(3) x(4)]);% set the position of the figure to the length and width of the axes
set(gca,'units','normalized','position',[0 0 1 1]); % set the axes units to pixels
set(gcf,'menubar','none');
% Lidar
blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast');
blueprint.set_attribute('points_per_second', '56000');
blueprint.set_attribute('range', '5000');
blueprint.set_attribute('sensor_tick', '0.1');
transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
lidar = world.spawn_actor(blueprint, transform, pyargs('attach_to',tesla));

moduleLidar = sensorBind(lidar, 'lidar_file', 'lidar', 'array');

% Spawn NPC's
npc_bps = blueprint_library.filter("vehicle");
npc_to_spawn = 75;

% Preallocate memory
npc_list = cell(1, npc_to_spawn);

i = 1;
while i <= npc_to_spawn
    try
        npc_bp = py.random.choice(npc_bps);
        spawn_point = py.random.choice(world.get_map().get_spawn_points());
        npc_list{i} = world.spawn_actor(npc_bp, spawn_point);
        npc_list{i}.set_autopilot(true);
    catch
        % In case spawing fails due to collision, try again
        i = i - 1;
    end
    i = i + 1;
end

% A bounding box detector model.
detectorModel = HelperBoundingBoxDetector(...
    'XLimits',[-50 75],...              % min-max
    'YLimits',[-5 5],...                % min-max
    'ZLimits',[-2 5],...                % min-max
    'SegmentationMinDistance',1.6,...   % minimum Euclidian distance
    'MinDetectionsPerCluster',1,...     % minimum points per cluster
    'MeasurementNoise',eye(6),...       % measurement noise in detection report
    'GroundMaxDistance',0.3);           % maximum distance of ground points from ground plane


assignmentGate = [10 100]; % Assignment threshold;
confThreshold = [7 10];    % Confirmation threshold for history logic
delThreshold = [8 10];     % Deletion threshold for history logic
Kc = 1e-5;                 % False-alarm rate per unit volume

% IMM filter initialization function
filterInitFcn = @helperInitIMMFilter;

% A joint probabilistic data association tracker with IMM filter
tracker = trackerJPDA('FilterInitializationFcn',filterInitFcn,...
    'TrackLogic','History',...
    'AssignmentThreshold',assignmentGate,...
    'ClutterDensity',Kc,...
    'ConfirmationThreshold',confThreshold,...
    'DeletionThreshold',delThreshold,...
    'HasDetectableTrackIDsInput',true,...
    'InitializationThreshold',0);

% Create display
displayObject = HelperLidarExampleDisplay(uint8(py.getattr(pyModule, 'array')),...
    'PositionIndex',[1 3 6],...
    'VelocityIndex',[2 4 7],...
    'DimensionIndex',[9 10 11],...
    'YawIndex',8,...
    'MovieName','',...  % Specify a movie name to record a movie.
    'RecordGIF',false); % Specify true to record new GIFs

%% Loop Through Data
% Loop through the recorded lidar data, generate detections from the
% current point cloud using the detector model and then process the
% detections using the tracker.
start_time = cputime;

% Initiate all tracks.
allTracks = struct([]);

% Rotate point cloud
A = [0 -1 0 0; ...
     1  0 0 0; ...
     0  0 1 0; ...
     0  0 0 1];
tform = affine3d(A);

while carla_is_running
    try
        % Update time
        time = cputime - start_time;

        % Get current lidar scan
        XYZI = single(py.getattr(moduleLidar, 'array'));
        XYZ = lidarData(:, 1:3);
        
        % Flip the axis
        XYZ(:, 2) = -1 * XYZ(:, 2);
        currentLidar = pctransform(pointCloud(XYZ), tform);

        % Generator detections from lidar scan.
        [detections,obstacleIndices,groundIndices,croppedIndices] = detectorModel(currentLidar,time);

        % Calculate detectability of each track.
        detectableTracksInput = helperCalcDetectability(allTracks,[1 3 6]);

        % Pass detections to track.
        [confirmedTracks,tentativeTracks,allTracks] = tracker(detections,time,detectableTracksInput);

        % Get model probabilities from IMM filter of each track using
        % getTrackFilterProperties function of the tracker.
        modelProbs = zeros(2,numel(confirmedTracks));
        for k = 1:numel(confirmedTracks)
            c1 = getTrackFilterProperties(tracker,confirmedTracks(k).TrackID,'ModelProbabilities');
            modelProbs(:,k) = c1{1};
        end

        % Update display
        if isvalid(displayObject.PointCloudProcessingDisplay.ObstaclePlotter)
            % Get current image scan for reference image
            currentImage = uint8(py.getattr(pyModule, 'array'));

            set(imageHandle,'Cdata',currentImage);
            
            % Update display object
            displayObject(detections,confirmedTracks,currentLidar,obstacleIndices,...
                groundIndices,croppedIndices,[],modelProbs);
        end
    catch
        carla_is_running = false;
        close all;
    end
end

tesla.destroy();
sensor.destroy();
lidar.destroy();

for i=1:npc_to_spawn
   npc_list{i}.destroy(); 
end