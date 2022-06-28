import carla
import pygame

import random
import numpy as np
import math
import queue

from models.cat_model_vae import BisimModel
from envs.bounding_box import get_bounding_boxes, draw_bounding_boxes, get_matrix
import torch


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True


def draw_image(env, image, is_ready):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    temp_array = array.copy()
    # Ready mode
    if is_ready:
        temp_array[:,:,:-1] = (temp_array[:,:,:-1] / 4).astype(np.int)
    temp_array = temp_array[:, :, :3]
    temp_array = temp_array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(temp_array.swapaxes(0, 1))
    env.display.blit(image_surface, (0, 0))
    if is_ready:
        myfont = pygame.font.SysFont('Comic Sans MS', 24)
        text = myfont.render('Press the red button on steering wheel to begin', False, (255, 255, 255))
        env.display.blit(text, (int(image.width / 3) - 50, int(image.height / 2)))
    # Display wall during normal mode
    # if not is_ready:
    #     player_location = env.player.get_location()
    #     walls = [wall for wall in env.all_walls if
    #              (np.linalg.norm(np.array([wall.bounding_box.location.x, wall.bounding_box.location.y]) - np.array(
    #                  [player_location.x, player_location.y])) < 40.)]
    #     b_boxes = get_bounding_boxes(walls, env.camera_rgb, 'wall')
    #     draw_bounding_boxes(env.display, b_boxes, 'wall')
    # Display goal line
    debug = env.world.debug
    debug.draw_line(begin=carla.Location(x=env.goal_left_loc.x, y=env.goal_left_loc.y, z=1.0),
                    end=carla.Location(x=env.goal_right_loc.x, y=env.goal_right_loc.y, z=1.0),
                    thickness=2.0,
                    color=carla.Color(0, 255, 0))


def process_lidar(lidar_scan):
    other_data = []
    car_data, wall_data = np.zeros((19, 1000, 2)), np.zeros((19, 1000, 2))
    car_counts, wall_counts = np.zeros((19,)), np.zeros((19,))

    obstacle_type = np.full((19,), 2.0)  # Car=0, Wall=1, Nothing=2
    interval = np.pi / 18.
    for location in lidar_scan:
        if location.point.x > 0.0:
            # if location.object_tag == 17:  # Wall
            #     point_angle = np.arctan(location.point.y / location.point.x)
            #     if point_angle % interval < 0.2 * interval or point_angle % interval > 0.8 * interval:
            #         point_angle += (np.pi / 2.0)
            #         # Round up or down to the nearest angle interval
            #         if point_angle % interval < 0.2 * interval:
            #             point_angle -= (point_angle % interval)
            #         elif point_angle % interval > 0.8 * interval:
            #             point_angle += (interval - (point_angle % interval))
            #         # Append to respective id in sensor_data
            #         wall_data[int(np.round(point_angle / interval)), int(
            #             wall_counts[int(np.round(point_angle / interval))])] = [
            #             location.point.x, location.point.y]
            #         wall_counts[int(np.round(point_angle / interval))] += 1
            if location.object_tag == 9:  # Others
                other_data.append([location.point.x, location.point.y])
            if location.object_tag == 10:  # Cars
                point_angle = np.arctan(location.point.y / location.point.x)
                if point_angle % interval < 0.2 * interval or point_angle % interval > 0.8 * interval:
                    point_angle += (np.pi / 2.0)
                    # Round up or down to the nearest angle interval
                    if point_angle % interval < 0.2 * interval:
                        point_angle -= (point_angle % interval)
                    elif point_angle % interval > 0.8 * interval:
                        point_angle += (interval - (point_angle % interval))
                    # Append to respective id in sensor_data
                    car_data[int(np.round(point_angle / interval)), int(
                        car_counts[int(np.round(point_angle / interval))])] = [
                        location.point.x, location.point.y]
                    car_counts[int(np.round(point_angle / interval))] += 1
    car_data = np.sum(car_data, axis=-2)
    # wall_data = np.sum(wall_data, axis=-2)
    car_data = car_data / car_counts[:, None]
    # wall_data = wall_data / wall_counts[:, None]
    car_data = np.linalg.norm(car_data, ord=2, axis=-1)
    # wall_data = np.linalg.norm(wall_data, ord=2, axis=-1)
    car_data[np.argwhere(np.isnan(car_data))] = 40.
    # wall_data[np.argwhere(np.isnan(wall_data))] = 40.
    car_data /= 40.0
    # wall_data /= 40.0
    # for i in range(car_data.shape[0]):
    #     if car_data[i] < 1.0 and wall_data[i] == 1.0:
    #         obstacle_type[i] = 0
    #     if car_data[i] == 1.0 and wall_data[i] < 1.0:
    #         car_data[i] = wall_data[i]
    #         obstacle_type[i] = 1
    for i in range(car_data.shape[0]):
        if car_data[i] < 1.0:
            obstacle_type[i] = 0

    return car_data, obstacle_type


class CarlaEnv():
    def __init__(self,
                 render_display=True,
                 render_fog=True,
                 host="127.0.0.1",
                 port=2000,
                 tm_port=8000,
                 frame_skip=1,
                 image_resolution=(1280, 720),
                 model=None,
                 steering_agent=None):
        self.render_display = render_display
        self.ready_mode = True
        self.image_resolution = image_resolution
        self.frame_skip = frame_skip
        self.actor_list = []
        # Setup display
        if self.render_display:
            pygame.init()
            self.display = pygame.display.set_mode(
                self.image_resolution,
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()
        # Set up client
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)
        self.tm_port = tm_port
        # Create world
        self.world = self.client.load_world("Town04")
        self.map = self.world.get_map()
        # Remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()
        # Set weather
        if render_fog:
            self.world.set_weather(carla.WeatherParameters(cloudiness=100.0,
                                                           precipitation=0.0,
                                                           fog_density=100.0,
                                                           sun_altitude_angle=70.0,
                                                           fog_distance=0.0))
        # Spawn actors
        self.player = None
        self.resetPlayer()
        self.player_lane = self.map.get_waypoint(self.player.get_location()).lane_id
        self.actor_list.append(self.player)
        self.vehicles_list = []
        self.resetOtherVehicles()
        # Define all walls in the area
        self.all_walls = self.world.get_environment_objects(carla.CityObjectLabel.GuardRail)
        self.all_walls = [wall for wall in self.all_walls if
                     (wall.bounding_box.location.x < -350. and wall.bounding_box.location.y > 0.)]
        # Define goal line
        self.goal_left_wpt = self.map.get_waypoint_xodr(road_id=45, lane_id=-1, s=300.0)
        self.goal_right_wpt = self.goal_left_wpt.get_right_lane().get_right_lane().get_right_lane()
        self.goal_left_loc = self.goal_left_wpt.transform.location
        self.goal_right_loc = self.goal_right_wpt.transform.location
        self.all_goal_lanes_loc, wpt = [], self.goal_left_wpt
        for i in range(4):
            self.all_goal_lanes_loc.append(wpt.transform.location)
            wpt = wpt.get_right_lane()
        # Attach onboard camera
        if self.render_display:
            self.attachCamera()
            self.actor_list.append(self.camera_rgb)
        # Attach collision sensor
        self.collision_intensity = []
        self.attachCollisionSensor()
        self.actor_list.append(self.collision_sensor)
        # Attach lidar sensor
        self.attachLidarSensor()
        self.actor_list.append(self.lidar_sensor)
        # Initialize synchronous mode
        if self.render_display:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rgb, self.lidar_sensor, fps=20)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.lidar_sensor, fps=20)
        self.world.tick()
        self.steering_agent=steering_agent


        if model is None:
            self.model = BisimModel(action_shape=(2,))
            self.model.load_state_dict(torch.load('./current_model/model_latest.pt'))
        else:
            self.model = model

        self.latent_size = self.model.latent_size
        self.model = self.model.to("cpu")
        self.model.eval()

        self._env_state = None
        self._env_description = None
        self._latent_state_env = None
        self._latent_state_agent = None
        self._state = None
        self._action_mask = None
        self._action = None

        # Run a while to unlock spawning lag
        for i in range(50):
            vehicle_control = carla.VehicleControl(
                throttle=1.0, steer=0.0, brake=0.0, hand_brake=False,
                reverse=False, manual_gear_shift=False)
            self.player.apply_control(vehicle_control)
            self.world.tick()
        self.resetPlayer()
        self.resetOtherVehicles()
        self.world.tick()

    def resetPlayer(self):
        lane = int(np.random.choice([-1, -2, -3, -4]))
        spawn_point = self.map.get_waypoint_xodr(road_id=41, lane_id=lane, s=0.0)
        transform = spawn_point.transform
        transform.location.z += 0.1
        if self.player is None:
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            blueprint.set_attribute('role_name', 'hero')
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
                self.player = self.world.try_spawn_actor(blueprint, transform)
        else:
            for i in range(20):
                vehicle_control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=False,
                    reverse=False, manual_gear_shift=False)
                self.player.apply_control(vehicle_control)
                self.world.tick()
            self.player.set_transform(transform)

    def resetOtherVehicles(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []
        # Setup traffic manager
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)  # 8000? which port?
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.set_synchronous_mode(True)
        # Blueprints
        blueprints = self.world.get_blueprint_library().filter('vehicle.tesla.model3')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # Spawn vehicles
        next_waypoint = self.map.get_waypoint(self.player.get_location())
        num_vehicles = 25
        distance = 30.
        other_car_waypoints = []
        for _ in range(num_vehicles):
            next_waypoint = next_waypoint.next(distance)[0]
            lanes_waypoint = next_waypoint

            num_lanes = 1
            while lanes_waypoint.lane_id != -1:
                lanes_waypoint = lanes_waypoint.get_left_lane()
            # Count how many lanes are there (excluding exits)
            while True:
                if lanes_waypoint.right_lane_marking.type == carla.LaneMarkingType.Solid or \
                        lanes_waypoint.right_lane_marking.type == carla.LaneMarkingType.NONE:
                    break
                else:
                    lanes_waypoint = lanes_waypoint.get_right_lane()
                    num_lanes += 1
            # Randomize lane to spawn
            lane_id = -random.randint(1, num_lanes)
            # lane_id = -random.randint(2, 3)
            while next_waypoint.lane_id != lane_id:
                if next_waypoint.lane_id > lane_id:
                    next_waypoint = next_waypoint.get_right_lane()
                elif next_waypoint.lane_id < lane_id:
                    next_waypoint = next_waypoint.get_left_lane()
            other_car_waypoints.append(next_waypoint)

        # barricade_wpt = self.map.get_waypoint(carla.Location(x=-501., y=258., z=2.), project_to_road=False, lane_type=carla.LaneType.Any)
        # barricade2_wpt = self.map.get_waypoint(carla.Location(x=-500.5, y=267.5, z=2.), project_to_road=False,
        #                                       lane_type=carla.LaneType.Any)
        # other_car_waypoints.append(barricade_wpt)
        # other_car_waypoints.append(barricade2_wpt)

        batch = []
        for n, waypoint in enumerate(other_car_waypoints):
            transform = waypoint.transform
            transform.location.z += 0.1
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))

        id = 0
        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)
            vehicle = self.world.get_actor(response.actor_id)
            speed_diff = np.random.uniform(25., 35.)
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)
            self.traffic_manager.auto_lane_change(vehicle, False)
            id += 1
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                self.vehicles_list.append(response.actor_id)

        # traffic_manager.global_percentage_speed_difference(30.0)

    def attachCamera(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.image_resolution[0]))
        bp.set_attribute('image_size_y', str(self.image_resolution[1]))
        self.camera_rgb = self.world.spawn_actor(
            bp,
            # carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
            carla.Transform(carla.Location(x=0.2, y=-0.2,z=1.17), carla.Rotation(pitch=0.0)),
            attach_to=self.player,
            attachment_type=carla.AttachmentType.Rigid)
        VIEW_FOV = 90
        calibration = np.identity(3)
        calibration[0, 2] = self.image_resolution[0] / 2.0
        calibration[1, 2] = self.image_resolution[1] / 2.0
        calibration[0, 0] = calibration[1, 1] = self.image_resolution[0] / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera_rgb.calibration = calibration

    def attachCollisionSensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp,
                                                       carla.Transform(),
                                                       attach_to=self.player)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_intensity.append(intensity)

    def attachLidarSensor(self):
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        bp.set_attribute('range', '40')
        bp.set_attribute('upper_fov', '0')
        bp.set_attribute('lower_fov', '-25')
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('points_per_second', '100000')
        # bp.set_attribute('atmosphere_attenuation_rate', '0.0')
        self.lidar_sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.2, z=1.1),
                            # carla.Transform(carla.Location(z=1.7),
                            carla.Rotation()),
            attach_to=self.player)

    # def random_shooting(self, next_control=None):
    #     batch_size = 32
    #     time_length = 2
    #     frame_skip = 3
    #     action_mask_list = []
    #     for i in range(time_length):
    #         action_mask = torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 19)
    #         action_mask = torch.cat([action_mask, torch.ones(size=(frame_skip, batch_size, 5))], dim=-1)
    #         action_mask_list += [action_mask]
    #     action_mask = torch.cat(action_mask_list, dim=0)
    #     current_state = torch.tensor(self._state).unsqueeze(dim=0).unsqueeze(dim=0)
    #     current_state = current_state.repeat(1, batch_size, 1)
    #     # print(current_state.shape)
    #     # print(action_mask)
    #
    #     accumulated_q_value = self.prediction_mask(current_state, action_mask, next_control)
    #     max_id = torch.argmax(accumulated_q_value, dim=1)
    #
    #     opti_action_mask = action_mask[:, max_id:max_id+1]
    #     return opti_action_mask
    def adpative_random_shooting(self):
        batch_size = 2000
        num_itr = 4
        time_length = 2
        frame_skip = 3
        action_mask_list = []
        for i in range(time_length):
            action_mask = []
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 0 1 2
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 3 4 5
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 6 7 8
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 1)]  # 9
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 10 11 12
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 13 14 15
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 16 17 18

            action_mask = torch.cat(action_mask + [torch.ones(size=(frame_skip, batch_size, 5))], dim=-1)
            action_mask_list += [action_mask]
        action_mask = torch.cat(action_mask_list, dim=0)
        current_state = torch.tensor(self._state).unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(1, batch_size, 1)
        # print(current_state.shape)
        # print(action_mask)

        p = 0.3
        decay = 0.8
        for i in range(num_itr):
            accumulated_q_value = self.prediction_mask(current_state, action_mask)
            max_id = torch.argmax(accumulated_q_value, dim=0)
            opti_action_mask = action_mask[:, max_id:max_id + 1]

            top_k_ids = torch.topk(accumulated_q_value, dim=0, k=int(batch_size/2)).indices.squeeze(dim=-1)
            action_mask = torch.index_select(action_mask, dim=1, index=top_k_ids)
            noise_mag = (torch.rand(size=action_mask.size()) < p).type(torch.float)
            noise_sign = 2.0 * ((torch.rand(size=action_mask.size()) > 0.5).type(torch.float) - 0.5)
            action_mask_rsample = ((action_mask + noise_mag * noise_sign)> 0.5).type(torch.float)
            action_mask = torch.cat([action_mask, action_mask_rsample], dim=1)
            p *= decay

        return opti_action_mask

    def random_shooting(self, next_control=None):
        batch_size = 20000
        time_length = 2
        frame_skip = 3
        action_mask_list = []
        for i in range(time_length):
            action_mask = []
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)] # 0 1 2
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 3 4 5
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 6 7 8
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 1)]  # 9
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 10 11 12
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 13 14 15
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 16 17 18

            action_mask = torch.cat(action_mask+[torch.ones(size=(frame_skip, batch_size, 5))], dim=-1)
            action_mask_list += [action_mask]
        action_mask = torch.cat(action_mask_list, dim=0)
        current_state = torch.tensor(self._state).unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(1, batch_size, 1)
        # print(current_state.shape)
        # print(action_mask)

        accumulated_q_value = self.prediction_mask(current_state, action_mask, next_control)
        # print(accumulated_q_value.size())
        max_id = torch.argmax(accumulated_q_value, dim=0)

        opti_action_mask = action_mask[:, max_id:max_id+1]
        return opti_action_mask

    def prediction_mask(self, current_state: torch.Tensor, action_mask: torch.Tensor, next_control:torch.Tensor=None):
        # current_state (1, batch, 2 * latent_size)
        # action_mask (time, batch, action (19+5))
        current_agent_state = current_state[...,:self.latent_size]
        current_env_state = current_state[...,self.latent_size:]

        time_length, batch_size, _ = action_mask.size()

        next_env_state = current_env_state
        next_agent_state = current_agent_state

        accumulated_q_value = 0.0
        discount_factor = 0.96
        for t in range(time_length):
            # given z_t, a_m_t, predict z_t+1
            # predict next env state
            next_env_state, agent_action = self.prediction(next_env_state, next_agent_state)

            # update human state
            # print(self._action_mask.shape)
            next_env_description = self.model.autoencoder.get_description_dist_normal(next_env_state).mean
            action_mask_t = action_mask[t:t + 1]
            #
            next_env_description_fake = torch.ones_like(next_env_description).type(torch.float)

            next_agent_state = self.model.autoencoder.get_latent_state_dist(
                description=next_env_description * action_mask_t,
                pre_state=next_agent_state,
                action=agent_action,
                mask=action_mask_t).sample()

            # next_env_description[..., 0:19] = action_mask_t[..., 0:19] * next_env_description[..., 0:19]\
            #                               + (1 - action_mask_t[..., 0:19]) * next_env_description_fake[..., 0:19]
            # # action_mask_torch[0:19] = 1.0
            # action_mask_fake_torch = torch.ones_like(action_mask_t).type(torch.float)
            # next_agent_state = self.model.autoencoder.get_latent_state_dist(
            #     description=next_env_description * action_mask_fake_torch,
            #     pre_state=next_agent_state,
            #     action=agent_action,
            #     mask=action_mask_fake_torch).sample()

            # get Q value

            mask_cost = torch.sum(action_mask_t, dim=-1, keepdim=True) * 0.7
            accumulated_q_value += discount_factor * (self.get_q_value(next_env_state)-mask_cost)
        if next_control is not None:
            accumulated_q_value -= torch.abs(action_mask[0] - next_control.unsqueeze(dim=0)).sum(dim=-1, keepdim=True) * 0.0
        return accumulated_q_value.squeeze(dim=0) # (batch, 1)

    def get_q_value(self, latent_state: torch.Tensor):
        _, policy_dist = self.model.policy(latent_state)
        policy_action = torch.tanh(policy_dist.mean)
        combined_feat = torch.cat([latent_state, policy_action], dim=-1)
        return self.model.qf1_model(combined_feat).mean

    def prediction(self, current_env_state: torch.Tensor, current_agent_state: torch.Tensor):
        # human take an action based current agent state
        action, action_agent_dist = self.model.policy(current_agent_state)
        action = torch.tanh(action)

        # forward environment based on current env state
        next_state_dist = self.model.autoencoder.transition(torch.cat([current_env_state, action], dim=-1))
        return next_state_dist.mean, action

    def get_current_env_state(self):
        return torch.tensor(self._latent_state_env)

    def get_current_agent_state(self):
        return torch.tensor(self._latent_state_agent)

    def reset(self):
        self.resetPlayer()
        self.world.tick()
        self.resetOtherVehicles()
        self.world.tick()
        self.collision_intensity = []
        obs, reward, collided, done = self.simulator_step(action=None)

        self._env_state = obs.copy()
        self._env_description = self.get_env_description(self._env_state).copy()
        env_state_agent = self._env_state
        env_description_agent = self.get_env_description(env_state_agent)

        env_state_torch = torch.tensor(self._env_state).type(torch.float)
        self._latent_state_env = self.model.get_state_representation(observation=env_state_torch)

        env_description_agent_torch = torch.tensor(env_description_agent).type(torch.float)
        env_description_agent_torch[:19] = 1.0
        mask = torch.ones(size=env_description_agent_torch.size()).type(torch.float)
        self._action_mask = mask.detach().numpy().copy()
        # self._action_mask = [self._action_mask[:, :, 0:1], self._action_mask[:, :, 0:1], self._action_mask[:, :, 0:1],
        #                      self._action_mask[:, :, 1:2], self._action_mask[:, :, 1:2], self._action_mask[:, :, 1:2],
        #                      self._action_mask[:, :, 2:3], self._action_mask[:, :, 2:3], self._action_mask[:, :, 2:3],
        #                      self._action_mask[:, :, 3:4], self._action_mask[:, :, 3:4], self._action_mask[:, :, 3:4],
        #                      self._action_mask[:, :, 4:5], self._action_mask[:, :, 4:5], self._action_mask[:, :, 4:5],
        #                      self._action_mask[:, :, 5:6], self._action_mask[:, :, 5:6], self._action_mask[:, :, 5:6],
        #                      self._action_mask[:, :, 5:6]]

        self._latent_state_agent = self.model.autoencoder.get_latent_state_dist(
            description=env_description_agent_torch * mask,
            pre_state=None,
            action=None,
            mask=mask).sample()
        self._action, action_agent_dist = self.model.policy(self._latent_state_agent.unsqueeze(dim=0))
        self._action = torch.tanh(self._action[0])
        self._latent_state_agent = self._latent_state_agent.detach().numpy()
        self._latent_state_env = self._latent_state_env.detach().numpy()

        self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        return self._state.copy(), collided

    def resetToCurrentPosition(self):
        curr_wpt = self.map.get_waypoint(self.player.get_location())
        respawn_wpt = curr_wpt.previous(10.)[0]
        transform = respawn_wpt.transform
        for i in range(20):
            vehicle_control = carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=False,
                reverse=False, manual_gear_shift=False)
            self.player.apply_control(vehicle_control)
            self.world.tick()
        self.player.set_transform(transform)
        obs, reward, collided, done = self.simulator_step(action=None)

        self._env_state = obs.copy()
        self._env_description = self.get_env_description(self._env_state).copy()
        env_state_agent = self._env_state
        env_description_agent = self.get_env_description(env_state_agent)

        env_state_torch = torch.tensor(self._env_state).type(torch.float)
        self._latent_state_env = self.model.get_state_representation(observation=env_state_torch)

        env_description_agent_torch = torch.tensor(env_description_agent).type(torch.float)
        env_description_agent_torch[:19] = 1.0
        mask = torch.ones(size=env_description_agent_torch.size()).type(torch.float)
        self._action_mask = mask.detach().numpy().copy()

        self._latent_state_agent = self.model.autoencoder.get_latent_state_dist(
            description=env_description_agent_torch * mask,
            pre_state=None,
            action=None,
            mask=mask).sample()
        self._action, action_agent_dist = self.model.policy(self._latent_state_agent.unsqueeze(dim=0))
        self._action = torch.tanh(self._action[0])
        self._latent_state_agent = self._latent_state_agent.detach().numpy()
        self._latent_state_env = self._latent_state_env.detach().numpy()

        self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        return self._state.copy(), collided

        return obs

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

    def get_obs(self):
        obs = np.zeros(4)
        vehicle_yaw = self.player.get_transform().rotation.yaw
        road_waypoint = self.map.get_waypoint(self.player.get_location(),
                                              project_to_road=True,
                                              lane_type=carla.LaneType.Driving)
        road_yaw = road_waypoint.transform.rotation.yaw
        if vehicle_yaw < 0:
            vehicle_yaw += 360.0
        if road_yaw < 0:
            road_yaw += 360.0
        # Normalized orientation difference
        obs[0] = road_yaw - vehicle_yaw
        if obs[0] <= -180.0:
            obs[0] += 360.0
        elif obs[0] > 180.0:
            obs[0] -= 360.0
        obs[0] /= 180.0

        dist, vel_s, speed, done = self.dist_from_center_lane()
        # Normalized velocity
        obs[1] = (vel_s * 3.6) / 40.
        # Normalized unsigned dist from center of lane
        obs[2] = dist / 1.75
        # Normalized dist from center of road (-ve: left, +ve: right)
        dist_from_center_road = 0.
        lane_id = road_waypoint.lane_id
        if lane_id == -1:
            dist_from_center_road = -3.5 - 1.75
            if dist > 1.75:
                dist_from_center_road = -3.5 - 1.75 - dist
        elif lane_id == -2:
            dist_from_center_road = -1.75
        elif lane_id == -3:
            dist_from_center_road = 1.75
        elif lane_id == -4:
            dist_from_center_road = 3.5 + 1.75
            if dist > 1.75:
                dist_from_center_road = 3.5 + 1.75 + dist
        obs[3] = dist_from_center_road / 10.
        # Angle of a location ahead
        obs = np.append(obs, self.next_few_waypoints())
        # print(lane_id, obs[3])
        return obs

    def next_few_waypoints(self):
        vehicle_location = self.player.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_s = vehicle_waypoint.s

        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        assert road_id is not None
        lane_id = int(vehicle_waypoint_closest_to_road.lane_id)
        goal_lane_id = lane_id

        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        if goal_waypoint is None:
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id,
                                                           vehicle_s + carla_waypoint_discretization)
        next_goal_waypoint1 = goal_waypoint.next(10.0)
        # next_goal_waypoint2 = goal_waypoint.next(10.0)
        curr_yaw = vehicle_waypoint_closest_to_road.transform.rotation.yaw
        if curr_yaw < 0:
            curr_yaw += 360.0
        if next_goal_waypoint1[0].transform.rotation.yaw < 0:
            next_goal_waypoint1[0].transform.rotation.yaw += 360.0
        # if next_goal_waypoint2[0].transform.rotation.yaw < 0:
        #     next_goal_waypoint2[0].transform.rotation.yaw += 360.0
        forward_locations = np.array([next_goal_waypoint1[0].transform.rotation.yaw - curr_yaw])
        # next_goal_waypoint2[0].transform.rotation.yaw - curr_yaw])
        for i in range(forward_locations.shape[0]):
            if forward_locations[i] <= -180.0:
                forward_locations[i] += 360.0
            elif forward_locations[i] > 180.0:
                forward_locations[i] -= 360.0
        forward_locations /= 180.0
        return forward_locations

    def get_env_description(self, env_state):
        return env_state.copy()

    def step_communication(self, action):
        rewards = []
        collideds = 0.0
        for _ in range(self.frame_skip):
            self._state, reward, collided, done = self.forward(self._state, action)
            rewards.append(reward)
            if collided != 0.0:
                collideds = 1.0
        reward = np.mean(rewards)

        return self._state.copy(), reward, collideds, done

    def forward(self, current_state: np.array, action_mask: np.array):
        # Update the latent state first, then sample action to forward environment
        # print(action_mask[1:7])
        # self._env_state = self._env_state.copy()
        # # self._env_state[:19] = 1.0
        # self._env_description = self.get_env_description(self._env_state)
        #
        # current_state_torch = torch.tensor(current_state).type(torch.float)
        # self._action_mask = action_mask.copy()
        #
        # # self._action_mask = [self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 0 1 2
        # #                      self._action_mask[1:2], self._action_mask[1:2], self._action_mask[1:2],  # 3 4 5
        # #                      self._action_mask[2:3], self._action_mask[2:3], self._action_mask[2:3],  # 6 7 8
        # #                      self._action_mask[3:4],  # 9
        # #                      self._action_mask[4:5], self._action_mask[4:5], self._action_mask[4:5],  # 10 11 12
        # #                      self._action_mask[5:6], self._action_mask[5:6], self._action_mask[5:6],  # 13 14 15
        # #                      self._action_mask[6:7], self._action_mask[6:7], self._action_mask[6:7],  # 16 17 18
        # #                      self._action_mask[7:]]
        # # self._action_mask = [self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 0 1 2
        # #                      self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 3 4 5
        # #                      self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 6 7 8
        # #                      self._action_mask[0:1],  # 9
        # #                      self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 10 11 12
        # #                      self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 13 14 15
        # #                      self._action_mask[0:1], self._action_mask[0:1], self._action_mask[0:1],  # 16 17 18
        # #                      self._action_mask[1:]]
        # # self._action_mask = np.concatenate(self._action_mask)
        # # print(self._action_mask.shape)
        # action_mask_torch = torch.tensor(self._action_mask).type(torch.float)  # .to('cpu')
        #
        # env_description_torch = torch.tensor(self._env_description).type(torch.float)
        # #
        # # env_description_fake_torch = torch.ones_like(env_description_torch).type(torch.float)
        # # env_description_torch[0:19] = action_mask_torch[0:19] * env_description_torch[0:19]\
        # #                               + (1 - action_mask_torch[0:19]) * env_description_fake_torch[0:19]
        # # # action_mask_torch[0:19] = 1.0
        # # action_mask_fake_torch = torch.ones_like(action_mask_torch).type(torch.float)
        # # self._latent_state_agent = self.model.autoencoder.get_latent_state_dist(
        # #     description=env_description_torch * action_mask_fake_torch,
        # #     pre_state=current_state_torch[:self.latent_size],
        # #     action=self._action,
        # #     mask=action_mask_fake_torch).sample()
        # #     mask=action_mask_fake_torch).sample()
        # self._latent_state_agent = self.model.autoencoder.get_latent_state_dist(
        #     description=env_description_torch * action_mask_torch,
        #     pre_state=current_state_torch[:self.latent_size],
        #     action=self._action,
        #     mask=action_mask_torch).sample()
        #
        # env_state_torch = torch.tensor(self._env_state).type(torch.float)
        # latent_state_env_torch = torch.tensor(self._latent_state_env).type(torch.float)
        # self._latent_state_env = self.model.get_state_representation(observation=env_state_torch,
        #                                                              pre_state=latent_state_env_torch,
        #                                                              action=self._action,
        #                                                              mask=None)
        # # Physical action using the learnt basic model
        # self._action, action_agent_dist = self.model.policy(self._latent_state_agent.unsqueeze(dim=0))
        # self._action = torch.tanh(self._action[0])
        # Physical action using the steering wheel
        # self._action = self.steering_agent.parseVehicleWheel()
        # self._action = torch.tensor([self._action.throttle, self._action.steer])
        # Physical action using the keyboard
        self._action = self.steering_agent.parseVehicleKey(self.clock.get_time())
        self._action = torch.tensor([self._action.throttle, self._action.steer])

        # print("env:", torch.tanh(self.model.policy(self._latent_state_env.unsqueeze(dim=0))[0]))
        # print("agent:", torch.tanh(self.model.policy(self._latent_state_agent.unsqueeze(dim=0))[0]))

        # self._action = self._action.detach().numpy().copy()

        # self._latent_state_agent = self._latent_state_agent.detach().numpy()
        # self._latent_state_env = self._latent_state_env.detach().numpy()
        # self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        # add communication cost to reward
        # print(reward)
        for _ in range(3):
            self._env_state, reward, collided, done = self.simulator_step(self._action.detach().numpy().copy())
        # self._env_state, reward, collided, done = self.simulator_step(self._action.detach().numpy().copy())
        lidar_type = self._env_state[19:-3]

        reward -= np.sum(self._action_mask[0:19]) * 0.25 * 1 / 3
        # print(np.sum(self._action_mask[0:19]) * 0.25 * 1 / 3)
        # print(action_mask_torch[:19].detach().numpy())
        return self._state, reward, collided, done

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            next_obs, reward, collided, done = self.simulator_step(action)
        return next_obs, reward, collided, done

    def simulator_step(self, action):
        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        if action is not None:
            throttle_brake = float(action[0])
            steer = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake
            vehicle_control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False)
            vehicle_velocity = self.player.get_velocity()
            vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
            speed = np.linalg.norm(vehicle_velocity_xy)
            if speed * 3.6 >= 40.0:
                vehicle_control.throttle = 0.0
            self.player.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0., 0., 0.

        for i in range(len(self.vehicles_list)):
            vehicle = self.world.get_actor(self.vehicles_list[i])
            if vehicle.get_speed_limit() == 90.0:
                speed_diff = np.random.uniform(100. - ((100. - 25.) / 3.),
                                               100. - ((100. - 35.) / 3.))
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)

        # Advance the simulation and wait for sensors data
        if self.render_display:
            snapshot, rgb_image, lidar_scan = self.sync_mode.tick(timeout=2.0)
        else:
            snapshot, lidar_scan = self.sync_mode.tick(timeout=2.0)

        lidar_data, lidar_type = process_lidar(lidar_scan)
        # Render display
        if self.render_display:
            draw_image(self, rgb_image, self.ready_mode)
            # if self._action_mask is not None:
            #     self.visualizeCars(lidar_data, lidar_type)
            pygame.display.flip()
        next_obs = self.get_obs()
        next_obs = np.concatenate((lidar_data, next_obs), axis=0)
        reward, done, collided = self.getReward(steer, brake, lidar_data, \
                                                lidar_type, next_obs[-2])
        # Check if ego car has reached goal line
        player_loc = self.player.get_location()
        for goal_loc in self.all_goal_lanes_loc:
            if np.linalg.norm(np.array([(goal_loc.x - player_loc.x, goal_loc.y - player_loc.y)])) < 2.0:
                done = True
        return next_obs, reward, collided, done

    def getReward(self, steer, brake, lidar_scan, lidar_type, dist_from_road_center):
        dist_from_center, vel_s, speed, done = self.dist_from_center_lane()
        vel_s_kmh = vel_s * 3.6
        collision_intensity = sum(self.collision_intensity)
        self.collision_intensity.clear()  # clear it ready for next time step
        assert collision_intensity >= 0.
        colliding = float(collision_intensity > 0.)
        vehicle_location = self.player.get_location()
        if colliding:
            done, reward = True, -10.
        else:
            lidar_colliding_cost = 0.0
            mid_beam_index = int((lidar_scan.shape[0] - 1) / 2.)
            for i in range(lidar_scan.shape[0]):
                if i == mid_beam_index and lidar_scan[i] < 0.5 and lidar_type[i] == 0:
                    lidar_colliding_cost = 2.
                if (i == mid_beam_index - 1 or i == mid_beam_index + 1) and \
                        lidar_scan[i] < 0.25 and lidar_type[i] == 0:
                    lidar_colliding_cost = 2.
                if (i == mid_beam_index - 2 or i == mid_beam_index + 2) and \
                        lidar_scan[i] < 0.15 and lidar_type[i] == 0:
                    lidar_colliding_cost = 2.
                if (mid_beam_index - 4 <= i <= mid_beam_index - 3 or mid_beam_index + 3 <= i <= mid_beam_index + 4) and \
                        lidar_scan[i] < 0.09 and lidar_type[i] == 0:
                    lidar_colliding_cost = 2.
                if lidar_scan[i] < 0.05 and lidar_type[i] == 0:
                    lidar_colliding_cost = 4.
                # if lidar_scan[i] < 0.075 and lidar_type[i] == 1:
                #     lidar_colliding_cost = 4.
            if dist_from_road_center > 0.7 or dist_from_road_center < -0.7:
                lidar_colliding_cost = 4.

            player_lane = self.map.get_waypoint(self.player.get_location()).lane_id
            lane_change_cost = 0. if player_lane == self.player_lane else 1.0
            lane_center_cost = dist_from_center / 1.75
            self.player_lane = player_lane
            dt = 0.05
            # reward = vel_s * dt / (1. + 0.1*dist_from_center) - 0.1 * lidar_cost \
            #          - 0.1 * brake - 0.1 * abs(steer) - lidar_colliding_cost - wall_cost
            if vel_s_kmh <= 40.0:
                reward = vel_s_kmh / 40.0 - lidar_colliding_cost - 0.5 * brake - \
                         0.2 * abs(steer) - lane_change_cost - 1.0 * lane_center_cost
            else:
                reward = -(vel_s_kmh - 40.0) / 40.0 - lidar_colliding_cost - \
                         0.5 * brake - 0.2 * abs(steer) - lane_change_cost - \
                         1.0 * lane_center_cost
        # print(reward)

        return reward, done, colliding

    def dist_from_center_lane(self):
        vehicle_location = self.player.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_s = vehicle_waypoint.s
        vehicle_velocity = self.player.get_velocity()
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        lane_waypoint = self.map.get_waypoint(vehicle_location,
                                              project_to_road=True,
                                              lane_type=carla.LaneType.Driving)
        road_id = lane_waypoint.road_id
        assert road_id is not None
        lane_id = int(lane_waypoint.lane_id)
        goal_lane_id = lane_id
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        if goal_waypoint is None:
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id,
                                                       vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id,
                                                           vehicle_s + carla_waypoint_discretization)
        if goal_waypoint is None:
            done, dist, vel_s = True, 100., 0.
        else:
            goal_location = goal_waypoint.transform.location
            goal_xy = np.array([goal_location.x, goal_location.y])
            # goal_xy = np.array([vehicle_waypoint.transform.location.x, vehicle_waypoint.transform.location.y])
            dist = np.linalg.norm(vehicle_xy - goal_xy)

            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))
            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left.")
                done, vel_s = True, 0.
            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)
                done = False

            # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        # if vehicle_velocity.z > 1.:
        #     print("Episode done: vertical velocity too high ({}), usually a simulator glitch".format(
        #         vehicle_velocity.z))
        #     done = True
        # if vehicle_location.z > 0.5:
        #     print("Episode done: vertical velocity too high ({}), usually a simulator glitch".format(
        #         vehicle_location.z))
        #     done = True

        return dist, vel_s, speed, done

    def visualizeCars(self, lidar_data, lidar_type):
        player_location = self.player.get_location()
        vehicles = self.world.get_actors().filter('vehicle.*')
        cars = []
        player_angle = self.player.get_transform().rotation.yaw / 180. * np.pi
        vehicles = [vehicle for vehicle in vehicles if
                    (0. < np.linalg.norm(np.array([vehicle.get_location().x, vehicle.get_location().y]) - np.array(
                        [player_location.x, player_location.y])) < 40.)]
        for vehicle in vehicles:
            vehicle_loc_wrt_player = vehicle.get_location()
            vehicle_loc_wrt_player.x -= player_location.x
            vehicle_loc_wrt_player.y -= player_location.y
            vehicle_angle_wrt_world = np.arctan(vehicle_loc_wrt_player.y / (vehicle_loc_wrt_player.x + 1e-6))
            if vehicle_loc_wrt_player.x < 0. and vehicle_loc_wrt_player.y > 0.:
                vehicle_angle_wrt_world = np.pi + vehicle_angle_wrt_world
            elif vehicle_loc_wrt_player.x < 0. and vehicle_loc_wrt_player.y < 0.:
                vehicle_angle_wrt_world = -np.pi + vehicle_angle_wrt_world
            vehicle_angle_wrt_player = vehicle_angle_wrt_world - player_angle
            if -np.pi / 2. <= vehicle_angle_wrt_player <= np.pi / 2.:
                nearest_beam_index = int(np.round((vehicle_angle_wrt_player + np.pi / 2.) / np.pi * 18.))
                if 0 <= nearest_beam_index <= 18:
                    if self._action_mask[nearest_beam_index] == 1 and \
                            lidar_type[nearest_beam_index] == 0:
                        cars.append(vehicle)

                # beam_dist = lidar_data[i] * 40.0
                # beam_loc = self.player.get_location()
                # beam_loc.x += 2.2 * np.cos(car_angle)
                # beam_loc.y += 2.2 * np.sin(car_angle)
                # obj_angle = ((i / 18.) - 0.5) * np.pi + car_angle
                # beam_loc.x += beam_dist * np.cos(obj_angle)
                # beam_loc.y += beam_dist * np.sin(obj_angle)
                #
                # for car in vehicles:
                #     if np.linalg.norm(np.array([beam_loc.x, beam_loc.y]) - \
                #                       np.array([car.get_location().x, car.get_location().y])) < 1.0:
                #         cars.append(car)
        b_boxes = get_bounding_boxes(cars, self.camera_rgb, 'car')
        draw_bounding_boxes(self.display, b_boxes, 'car')
        # pygame.display.flip()
