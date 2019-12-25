clear;
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

while carla_is_running
    pause(1);
    clc;
    
    currentLocation = actor_list{id}.get_location();
    fprintf('Location: [m]\n')
    fprintf('x: %.2f \t y: %.2f \t z: %.2f \n\n',currentLocation.x, currentLocation.y, currentLocation.z);
    
    fprintf('Velocity: [m/s]\n')
    currentVelocity = actor_list{id}.get_velocity();
    fprintf('x: %.2f \t y: %.2f \t z: %.2f \n\n',currentVelocity.x, currentVelocity.y, currentVelocity.z);
    
    fprintf('Acceleration: [m/s^2]\n')
    currentAcceleration = actor_list{id}.get_acceleration();
    fprintf('x: %.2f \t y: %.2f \t z: %.2f \n\n',currentAcceleration.x, currentAcceleration.y, currentAcceleration.z);
    
    omega = char (hex2dec ( '03C9' ));
    fprintf('Angular Velocity: [%s]\n', omega);
    currentAngVelocity = actor_list{id}.get_angular_velocity();
    fprintf('x: %.2f \t y: %.2f \t z: %.2f \n\n',currentAngVelocity.x, currentAngVelocity.y, currentAngVelocity.z);
    
end