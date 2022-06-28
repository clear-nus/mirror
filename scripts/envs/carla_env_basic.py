import carla
import pygame
import cv2

import random
import numpy as np
import math
import queue

from envs.hud import HUD


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


def draw_combined_image(surface, image_front, image_back):
    front_array = np.frombuffer(image_front.raw_data, dtype=np.dtype("uint8"))
    back_array = np.frombuffer(image_back.raw_data, dtype=np.dtype("uint8"))
    front_array = np.reshape(front_array, (image_front.height, image_front.width, 4))
    back_array = np.reshape(back_array, (image_back.height, image_back.width, 4))
    front_array = front_array.copy()[:, :, :3]
    back_array = back_array.copy()[:, :, :3]
    front_array = front_array[int(image_front.height / 4):int(3 * image_front.height/4), :, :]
    back_array = back_array[int(image_back.height / 4):int(3 * image_back.height / 4), :, :]
    front_array = front_array[:, :, ::-1]
    back_array = back_array[:, :, ::-1]
    overall_array = np.concatenate((front_array, back_array), axis=0)

    image_surface = pygame.surfarray.make_surface(overall_array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))

def draw_front_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    temp_array = array.copy()
    temp_array = temp_array[:, :, :3]
    temp_array = temp_array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(temp_array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))

def draw_fpv_image(surface, image_front, image_back, render_fog):
    front_array = np.frombuffer(image_front.raw_data, dtype=np.dtype("uint8"))
    back_array = np.frombuffer(image_back.raw_data, dtype=np.dtype("uint8"))
    front_array = np.reshape(front_array, (image_front.height, image_front.width, 4))
    back_array = np.reshape(back_array, (image_back.height, image_back.width, 4))
    front_array = front_array.copy()
    # back_array = back_array.copy()
    if render_fog:
        front_array[:, :, :-1] = (front_array[:, :, :-1] / 1.2).astype(np.int)
    # back_array[:, :, :-1] = (2 * back_array[:, :, :-1] / 3).astype(np.int)
    front_array = front_array[:, :, :3]
    back_array = back_array[:, :, :3]
    front_array = front_array[:, :, ::-1]
    back_array = back_array[:, :, ::-1]

    if render_fog:
        blurred_temp_array = cv2.GaussianBlur(front_array[:int(26 * image_front.height / 48), :, :], (39, 39), 0)
        front_array = np.concatenate((blurred_temp_array, front_array[int(26 * image_front.height / 48):, :, :]), axis=0)

        back_array = cv2.GaussianBlur(back_array, (39, 39), 0)

    image_surface = pygame.surfarray.make_surface(front_array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))
    image_surface = pygame.surfarray.make_surface(back_array.swapaxes(0, 1))
    surface.blit(image_surface, (220, 30))#(370, 30))#(610, 60))

def process_lidar(lidar_scan, env):

    car_data, wall_data = np.zeros((36, 1000, 3)), np.zeros((36, 1000, 3))
    car_data_dheight = np.zeros((36, 1000, 3, 3))
    car_data_id_points, car_data_vel = np.zeros((36, 1000)), np.zeros((36))
    car_counts, wall_counts = np.zeros((36,)), np.zeros((36,))
    car_dheight_counts = np.zeros((36, 3))
    obstacle_type = np.full((36,), 2.0)  # Car=0, Wall=1, Nothing=2
    interval = np.pi / 18.0
    for point in lidar_scan:
        if point.object_tag == 10 and point.object_idx != env.player.id:
            # -180 <= angle < 180
            point_angle = np.arctan(abs(point.point.y / (point.point.x + 1e-6)))
            if point_angle % interval < 0.2 * interval or point_angle % interval > 0.8 * interval:
                if point.point.x <= 0.0 and point.point.y > 0.0:  # 90<=angle<180
                    point_angle = 2 * np.pi - point_angle
                elif point.point.x < 0.0 and point.point.y <= 0.0:  # -180<=angle<-90
                    point_angle = point_angle
                elif point.point.x >= 0.0 and point.point.y < 0.0:  # -90<=angle<0
                    point_angle = np.pi - point_angle
                elif point.point.x > 0.0 and point.point.y >= 0.0:  # 0<=angle<90
                    point_angle = np.pi + point_angle
                # Round up or down to the nearest angle interval
                if point_angle % interval < 0.2 * interval:
                    point_angle -= (point_angle % interval)
                elif point_angle % interval > 0.8 * interval:
                    point_angle += (interval - (point_angle % interval))
                # Append to respective id in sensor_data
                if int(np.round(point_angle / interval)) == 36:
                    car_data[0, int(car_counts[0])] = [point.point.x, point.point.y, point.point.z]
                    car_data_id_points[0, int(car_counts[0])] = point.object_idx
                    car_counts[0] += 1
                    # Discretize the points according to the height (0.0-0.5, 0.5-1.0, 1.0-1.7)
                    if point.point.z < -1.2:
                        car_data_dheight[0, int(car_dheight_counts[0, 0]), 0] = [point.point.x, point.point.y, point.point.z]
                        car_dheight_counts[0, 0] += 1
                    elif -1.2 <= point.point.z < -0.7:
                        car_data_dheight[0, int(car_dheight_counts[0, 1]), 1] = [point.point.x, point.point.y, point.point.z]
                        car_dheight_counts[0, 1] += 1
                    elif point.point.z >= -0.7:
                        car_data_dheight[0, int(car_dheight_counts[0, 2]), 2] = [point.point.x, point.point.y, point.point.z]
                        car_dheight_counts[0, 2] += 1
                else:
                    car_data[int(np.round(point_angle / interval)), int(
                        car_counts[int(np.round(point_angle / interval))])] = [
                        point.point.x, point.point.y, point.point.z]
                    car_data_id_points[int(np.round(point_angle / interval)), int(
                        car_counts[int(np.round(point_angle / interval))])] = point.object_idx
                    car_counts[int(np.round(point_angle / interval))] += 1
                    # Discretize the points according to the height (0.0-0.5, 0.5-1.0, 1.0-1.7)
                    if point.point.z < -1.2:
                        car_data_dheight[int(np.round(point_angle / interval)), int(
                        car_dheight_counts[int(np.round(point_angle / interval)), 0]), 0] = [point.point.x,
                                                                                             point.point.y,
                                                                                             point.point.z]
                        car_dheight_counts[int(np.round(point_angle / interval)), 0] += 1
                    elif -1.2 <= point.point.z < -0.7:
                        car_data_dheight[int(np.round(point_angle / interval)), int(
                            car_dheight_counts[int(np.round(point_angle / interval)), 1]), 1] = [point.point.x,
                                                                                                 point.point.y,
                                                                                                 point.point.z]
                        car_dheight_counts[int(np.round(point_angle / interval)), 1] += 1
                    elif point.point.z >= -0.7:
                        car_data_dheight[int(np.round(point_angle / interval)), int(
                            car_dheight_counts[int(np.round(point_angle / interval)), 2]), 2] = [point.point.x,
                                                                                                 point.point.y,
                                                                                                 point.point.z]
                        car_dheight_counts[int(np.round(point_angle / interval)), 2] += 1
    # car_data_dheight.shape is (36, 1000, 3, 3)
    car_data_dheight = np.sum(car_data_dheight, axis=-3)
    # car_data_dheight.shape is (36, 3, 3)
    car_data_dheight = car_data_dheight / car_dheight_counts[:, :, None]
    car_data_dheight = np.reshape(car_data_dheight, (car_data_dheight.shape[0] * car_data_dheight.shape[1], 3))
    # car_data_dheight.shape is (108, 3)
    car_data_dheight = np.linalg.norm(car_data_dheight, ord=2, axis=-1)
    # car_data_dheight.shape is (108,)
    car_data_dheight[np.argwhere(np.isnan(car_data_dheight))] = 40.
    car_data_dheight /= 40.0
    car_data_dheight = np.reshape(car_data_dheight, (36, 3))

    car_data = np.sum(car_data, axis=-2)
    car_data = car_data / car_counts[:, None]
    car_data = np.linalg.norm(car_data, ord=2, axis=-1)
    car_data[np.argwhere(np.isnan(car_data))] = 40.
    car_data /= 40.0
    for i in range(car_data.shape[0]):
        if car_data[i] < 1.0:
            obstacle_type[i] = 0

    # Get detected actor id for each beam
    car_data_id = np.zeros((36,))
    car_data_id_counts = np.expand_dims(np.count_nonzero(car_data_id_points == env.vehicles_list[0], axis=-1), axis=-1)
    for i in range(1, 4):
        car_data_id_counts = np.concatenate((car_data_id_counts,
                                             np.expand_dims(np.count_nonzero(car_data_id_points == env.vehicles_list[i], axis=-1), axis=-1)),
                                            axis=-1)
    car_data_id_argmax = np.argmax(car_data_id_counts, axis=-1)
    for i in range(car_data_id_counts.shape[0]):
        if np.count_nonzero(car_data_id_counts[i]) != 0:
            car_data_id[i] = env.vehicles_list[car_data_id_argmax[i]]
        else:
            car_data_id[i] = 0.0
    # Get vehicle speed for each beam
    for i in range(car_data_id.shape[0]):
        if car_data_id[i] > 0:
            vehicle = env.world.get_actor(int(car_data_id[i]))
            vel = vehicle.get_velocity()
            speed = np.linalg.norm(np.array([vel.x, vel.y])) * 3.6
            car_data_vel[i] = speed

    return car_data_dheight, car_data, car_data_id, car_data_vel, obstacle_type


class CarlaEnvBackCar():
    def __init__(self,
                 render_display=True,
                 render_fog=True,
                 host="127.0.0.1",
                 port=2000,
                 tm_port=8000,
                 frame_skip=1,
                 image_resolution=(1280, 720),
                 gpt_device="cuda:0"):
        self.render_display = render_display
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
            self.hud = HUD(self.image_resolution[0], self.image_resolution[1])
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
        self.render_fog = render_fog
        # Spawn actors
        self.player = None
        self.resetPlayer()
        self.player_lane = self.map.get_waypoint(self.player.get_location()).lane_id
        self.actor_list.append(self.player)
        self.vehicles_list = []
        self.resetOtherVehicles()
        # Attach onboard camera
        if self.render_display:
            self.attachCamera()
            self.actor_list.append(self.camera_rgb_front)
            self.actor_list.append(self.camera_rgb_back)
        # Attach collision sensor
        self.collision_intensity = []
        self.attachCollisionSensor()
        self.actor_list.append(self.collision_sensor)
        # Attach lidar sensor
        self.attachLidarSensor()
        self.actor_list.append(self.lidar_sensor)
        # Initialize synchronous mode
        if self.render_display:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rgb_front,
                                           self.camera_rgb_back, self.lidar_sensor, fps=20)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.lidar_sensor, fps=20)
        self.world.tick()

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
        self.slow_down_pos = float(random.randint(230, 300))

    def resetPlayer(self):
        # Reset scenario_mode
        player_lane = random.choice([-2, -3])
        self.speed_up_down = random.choice([1, 2, 3, 4])  # no-op/up-left/up-right/down-left/down-right
        # player_lane = random.choice([-2])
        # self.speed_up_down = random.choice([1])  # no-op/up-left/up-right/down-left/down-right
        if self.speed_up_down == 0:
            print('scenario: no-op')
        elif self.speed_up_down == 1:
            print('scenario: up-left')
        elif self.speed_up_down == 2:
            print('scenario: up-right')
        elif self.speed_up_down == 3:
            print('scenario: down-left')
        elif self.speed_up_down == 4:
            print('scenario: down-right')
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=player_lane, s=30.0)
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
        # Clear out old vehicles
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
        other_car_waypoints = []
        # Rear two cars
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=0.0)
        other_car_waypoints.append(spawn_point)
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-3, s=0.0)
        other_car_waypoints.append(spawn_point)
        # Front two cars
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=130.0)
        other_car_waypoints.append(spawn_point)
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-3, s=130.0)
        other_car_waypoints.append(spawn_point)

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
            if self.speed_up_down == 1:  # up-left
                if id == 0 or id == 2:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -120.)
                elif id == 1:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -35.0)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -64.0)
            elif self.speed_up_down == 2:  # up-right
                if id == 1 or id == 3:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -120.)
                elif id == 0:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -35.0)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -64.0)
            else:  # no-op
                if id == 0 or id == 1:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -35.0)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -64.0)
            self.traffic_manager.auto_lane_change(vehicle, False)
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
            id += 1
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                self.vehicles_list.append(response.actor_id)

        # traffic_manager.global_percentage_speed_difference(30.0)

    def attachCamera(self):
        bp_front = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_front.set_attribute('image_size_x', str(self.image_resolution[0]))
        bp_front.set_attribute('image_size_y', str(self.image_resolution[1]))
        bp_back = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_back.set_attribute('image_size_x', '360')
        bp_back.set_attribute('image_size_y', '55')
        self.camera_rgb_front = self.world.spawn_actor(
            bp_front,
            # carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
            carla.Transform(carla.Location(x=0.1, y=-0.0, z=1.12), carla.Rotation(pitch=0.0)),
            # carla.Transform(carla.Location(z=1.45), carla.Rotation(pitch=0.0)),
            # carla.Transform(carla.Location(x=-0.7, y=0.0, z=1.4), carla.Rotation(yaw=180.0)),
            attach_to=self.player,
            attachment_type=carla.AttachmentType.Rigid)
        self.camera_rgb_back = self.world.spawn_actor(
            bp_back,
            # carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
            # carla.Transform(carla.Location(x=0.2, y=-0.2, z=1.17), carla.Rotation(pitch=0.0)),
            carla.Transform(carla.Location(x=-1.0, y=0.0, z=1.4), carla.Rotation(yaw=180.0)),
            attach_to=self.player,
            attachment_type=carla.AttachmentType.Rigid)
        VIEW_FOV = 90
        calibration = np.identity(3)
        calibration[0, 2] = self.image_resolution[0] / 2.0
        calibration[1, 2] = self.image_resolution[1] / 2.0
        calibration[0, 0] = calibration[1, 1] = self.image_resolution[0] / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera_rgb_front.calibration = calibration

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
        bp.set_attribute('lower_fov', '-30')
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('points_per_second', '100000')
        # bp.set_attribute('atmosphere_attenuation_rate', '0.0')
        self.lidar_sensor = self.world.spawn_actor(
            bp,
            # carla.Transform(carla.Location(x=2.2, z=1.1),
            carla.Transform(carla.Location(z=1.7),
                            carla.Rotation()),
            attach_to=self.player)

    def reset(self):
        self.resetPlayer()
        self.world.tick()
        self.resetOtherVehicles()
        self.world.tick()
        self.collision_intensity = []
        self.slow_down_pos = float(random.randint(230, 300))
        obs, lidar_data_id, lidar_data_vel, reward, collided, done = self.simulator_step(action=None)
        return obs, collided, done

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
        if lane_id == -2:
            dist_from_center_road = -1.75
            if dist > 1.75:
                dist_from_center_road = -1.75 - dist
        elif lane_id == -3:
            dist_from_center_road = 1.75
            if dist > 1.75:
                dist_from_center_road = 1.75 + dist
        obs[3] = dist_from_center_road / 7.
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

    def step(self, action):
        rewards = []
        for _ in range(self.frame_skip):
            next_obs, lidar_data_id, lidar_data_vel, reward, collided, done = self.simulator_step(action)
            # print('test', next_obs[1])
            rewards.append(reward)
            if done or collided:
                break
        return next_obs, lidar_data_id, lidar_data_vel, np.mean(rewards), collided, done

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
            if speed * 3.6 >= 37.0:
                vehicle_control.throttle = 0.0
            self.player.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0., 0., 0.

        # Advance the simulation and wait for sensors data
        # if frame_count == 0 or frame_count == 1:
        if self.render_display:
            snapshot, self.rgb_image_front, self.rgb_image_back, self.lidar_scan = self.sync_mode.tick(timeout=2.0)
        else:
            snapshot, self.lidar_scan = self.sync_mode.tick(timeout=2.0)

        # Check whether the front cars have reached the desired place to stop and also
        # if the scenario is to slow down.
        if self.speed_up_down == 3:  # down-left
            checkpt = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=self.slow_down_pos).transform.location
            front_car = self.world.get_actor(self.vehicles_list[2])
            if np.linalg.norm(np.array([checkpt.x - front_car.get_location().x,
                                        checkpt.y - front_car.get_location().y])) < 5.0:
                back_car = self.world.get_actor(self.vehicles_list[0])
                vehicle_control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.1, hand_brake=False,
                    reverse=False, manual_gear_shift=False)
                front_car.apply_control(vehicle_control)
                back_car.apply_control(vehicle_control)
                self.traffic_manager.vehicle_percentage_speed_difference(front_car, 70.)
                self.traffic_manager.vehicle_percentage_speed_difference(back_car, 70.)
        elif self.speed_up_down == 4:  # down-right
            checkpt = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=self.slow_down_pos).transform.location
            front_car = self.world.get_actor(self.vehicles_list[3])
            if np.linalg.norm(np.array([checkpt.x - front_car.get_location().x,
                                        checkpt.y - front_car.get_location().y])) < 5.0:
                back_car = self.world.get_actor(self.vehicles_list[1])
                vehicle_control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.1, hand_brake=False,
                    reverse=False, manual_gear_shift=False)
                front_car.apply_control(vehicle_control)
                back_car.apply_control(vehicle_control)
                self.traffic_manager.vehicle_percentage_speed_difference(front_car, 90.)
                self.traffic_manager.vehicle_percentage_speed_difference(back_car, 90.)

        lidar_data_3d, lidar_data_2d, lidar_data_id, lidar_data_vel, lidar_type = process_lidar(self.lidar_scan, self)
        # for i in range(lidar_data_3d.shape[0]):
        #     print("{:.5f}".format(lidar_data_3d[i, 2]), end=" ")
        #     # print("{:.5f}, {:.5f}, {:.5f}".format(lidar_data_3d[17, 2], lidar_data_3d[18, 2], lidar_data_3d[19, 2]), '|', "{:.5f}, {:.5f}, {:.5f}".format(lidar_data_3d[1, 2], lidar_data_3d[0, 2], lidar_data_3d[-1, 2]))
        # print("")
        lidar_data_3d = np.reshape(lidar_data_3d, (lidar_data_3d.shape[0] * lidar_data_3d.shape[1],))
        # Render display
        if self.render_display:
            # draw_combined_image(self.display, rgb_image_front, rgb_image_back)
            # draw_front_image(self.display, rgb_image_front)
            draw_fpv_image(self.display, self.rgb_image_front, self.rgb_image_back, self.render_fog)
            self.hud.tick(self, self.clock)
            self.hud.render(self.display)
            pygame.display.flip()

        next_obs = self.get_obs()
        next_obs = np.concatenate((lidar_data_3d, next_obs), axis=0)
        reward, done, collided = self.getReward(steer, brake, lidar_data_2d, lidar_type, next_obs[-2])

        return next_obs, lidar_data_id, lidar_data_vel, reward, collided, done

    def getReward(self, steer, brake, lidar_scan, lidar_type, dist_from_road_center):
        dist_from_center, vel_s, speed, done = self.dist_from_center_lane()
        vel_s_kmh = vel_s * 3.6
        collision_intensity = sum(self.collision_intensity)
        self.collision_intensity.clear()  # clear it ready for next time step
        assert collision_intensity >= 0.
        colliding = float(collision_intensity > 0.)
        if colliding:
            done, reward = True, -10.
        else:
            front_lidar_cost, surround_lidar_cost, back_lidar_cost, wall_cost = 0.0, 0.0, 0.0, 0.0
            front_beam_index = int(lidar_scan.shape[0] / 2.)
            for i in range(lidar_scan.shape[0]):
                if i == front_beam_index and lidar_scan[i] < 0.5 and lidar_type[i] == 0:
                    front_lidar_cost = 2.
                # if (i == front_beam_index - 1 or i == front_beam_index + 1) and \
                #         lidar_scan[i] < 0.125 and lidar_type[i] == 0:
                #     front_lidar_cost = 2.
                if lidar_scan[i] < 0.04 and lidar_type[i] == 0:
                    surround_lidar_cost = 4.
                if i == 0 and lidar_scan[i] < 0.5 and lidar_type[i] == 0:
                    back_lidar_cost = 2.
            if dist_from_road_center > 0.5 or dist_from_road_center < -0.5:
                wall_cost = 4.

            player_lane = self.map.get_waypoint(self.player.get_location()).lane_id
            lane_change_cost = 0. if player_lane == self.player_lane else 1.0
            lane_center_cost = dist_from_center / 1.75
            self.player_lane = player_lane


            speed_reward = vel_s_kmh / 40. if vel_s_kmh <= 40. else -(vel_s_kmh - 40.) / 40.

            reward = speed_reward - 0.3 * brake - 0.1 * abs(steer) - \
                     1.0 * (lane_change_cost + lane_center_cost) - \
                     front_lidar_cost - surround_lidar_cost - back_lidar_cost - wall_cost
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
        if vehicle_velocity.z > 1.:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch".format(
                vehicle_velocity.z))
            done = True
        if vehicle_location.z > 0.5:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch".format(
                vehicle_location.z))
            done = True

        return dist, vel_s, speed, done
