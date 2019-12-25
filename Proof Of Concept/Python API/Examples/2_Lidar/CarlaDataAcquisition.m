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

% Lidar
blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast');
blueprint.set_attribute('points_per_second', '140000');
blueprint.set_attribute('range', '2500');
blueprint.set_attribute('sensor_tick', '0.1');
blueprint.set_attribute('upper_fov', '45.0')
blueprint.set_attribute('lower_fov', '-30.0')
transform = py.carla.Transform(py.carla.Location(pyargs('x',0.8, 'z',1.7)));
sensor = world.spawn_actor(blueprint, transform, pyargs('attach_to',tesla));

moduleLidar = sensorBind(sensor, 'lidar_file', 'lidar', 'array');

player = pcplayer([-25 25],[-25 25],[-10 10]);

while isOpen(player)
     view(player, single(py.getattr(moduleLidar, 'array')));
end

tesla.destroy();
sensor.destroy();