port = int16(2000);
client = py.carla.Client('localhost', port);
client.set_timeout(2.0);
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
blueprint.set_attribute('image_size_x', '1920')
blueprint.set_attribute('image_size_y', '1080')
% blueprint.set_attribute('sensor_tick', '.1');
transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
sensor = world.spawn_actor(blueprint, transform, pyargs('attach_to',tesla));

pyModule = sensorBind(sensor, "rgb", "rgb", "array");

% Display Initial Frame
image = py.getattr(pyModule, 'array');
imMat = uint8(image);

imageHandle = imshow(imMat);

% Handle keyboard
callstr = 'set(gcbf, ''Userdata'', get(gcbf, ''Currentkey'')) ; uiresume ' ;
keyBoardHandle = figure( 'name', 'Press a key', ...
                         'keypressfcn', callstr, ...
                         'windowstyle', 'modal', ...
                         'numbertitle', 'off', ...
                         'position', [0 0 1 1], ...
                         'userdata', 'timeout');
                     
time_last_update = cputime;

while carla_is_running
    try
         image = py.getattr(pyModule, 'array');
         imMat = uint8(image);

         % Write directly to Cdata. Consideribly faster than imshow
         set(imageHandle,'Cdata',imMat);
         drawnow;

         % Process keyboard
         keyPressed = get(keyBoardHandle,'Userdata');
         
         % Press escape to quit
         if strcmp(keyPressed, 'escape')
             carla_is_running = false;
             close all;
         end
         
         dt = cputime - time_last_update;
         
         processKeyboard(tesla, keyPressed, dt);
         
         % Reset the pressed key
         set(keyBoardHandle,'Userdata', char(0));
         
         clc;
         disp(keyPressed);
         disp(tesla.get_control());

         time_last_update = time_last_update + dt;
         
         pause(0.001);
    catch
       carla_is_running = false;
       close all;
    end
end

tesla.destroy();
sensor.destroy();