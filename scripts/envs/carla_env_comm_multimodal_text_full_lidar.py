import carla
import pygame

import random
import numpy as np
import math
import queue

from envs.hud import HUD

from models.basic_multimodal_text_cat_model import BisimModel
from envs.bounding_box import get_bounding_boxes, draw_bounding_boxes
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import cv2


def generate_text_embed(obs: torch.Tensor, tokenizer, gpt_model, gpt_device="cuda:0", device="cuda:1"):
    if obs[0] < 0.5:
        text = "The vehicle at the back is approaching our vehicle."
    elif 0.5 <= obs[0] < 1.0:
        text = "A vehicle detected at the back on our lane."
    else:
        text = "No vehicle detected at the back on our lane."
    text_token = tokenizer.encode(text)
    text_token = torch.tensor(text_token).to(gpt_device)
    text_length = text_token.size(0)

    gpt_embed = gpt_model.transformer(text_token.unsqueeze(dim=0).to(gpt_device))[0]
    gpt_embed = torch.flatten(gpt_embed, start_dim=-2, end_dim=-1)[0].detach().cpu().numpy()
    text_token = text_token.detach().cpu().numpy()
    # print(text)
    return gpt_embed, text_token


def generate_text(text_embed: torch.Tensor, gpt_predictor, tokenizer):
    gpt_embed_size = 768
    prediction = gpt_predictor(text_embed.reshape(-1, 10, gpt_embed_size))

    predicted_text = ""
    for i in range(10):
        predicted_index = torch.argmax(prediction[0, i, :]).item()
        predicted_text += tokenizer.decode(predicted_index)
    return predicted_text


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


def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def draw_image(surface, image_front, image_back):
    front_array = np.frombuffer(image_front.raw_data, dtype=np.dtype("uint8"))
    back_array = np.frombuffer(image_back.raw_data, dtype=np.dtype("uint8"))
    front_array = np.reshape(front_array, (image_front.height, image_front.width, 4))
    back_array = np.reshape(back_array, (image_back.height, image_back.width, 4))
    front_array = front_array.copy()[:, :, :3]
    back_array = back_array.copy()[:, :, :3]
    front_array = front_array[int(image_front.height / 4):int(3 * image_front.height / 4), :, :]
    back_array = back_array[int(image_back.height / 4):int(3 * image_back.height / 4), :, :]
    front_array = front_array[:, :, ::-1]
    back_array = back_array[:, :, ::-1]
    overall_array = np.concatenate((front_array, back_array), axis=0)

    # blurred_temp_array = cv2.GaussianBlur(temp_array, (21, 21), 0)
    # mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    # mask = cv2.rectangle(mask, (0, int(31 * image.height / 48)), (image.width, image.height), (255, 255, 255), -1)
    # temp_array = np.where(mask == (255, 255, 255), temp_array, blurred_temp_array)
    #
    # blurred_temp_array = cv2.GaussianBlur(temp_array, (49, 49), 0)
    # mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    # mask = cv2.rectangle(mask, (0, int(14 * image.height / 24)), (image.width, image.height), (255, 255, 255), -1)
    # temp_array = np.where(mask == (255, 255, 255), temp_array, blurred_temp_array)

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


def process_lidar(lidar_scan, env):
    car_data, wall_data = np.zeros((18, 1000, 2)), np.zeros((18, 1000, 2))
    car_counts, wall_counts = np.zeros((18,)), np.zeros((18,))
    obstacle_type = np.full((18,), 2.0)  # Car=0, Wall=1, Nothing=2
    interval = np.pi / 9.0
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
                if point_angle % interval < 0.1 * interval:
                    point_angle -= (point_angle % interval)
                elif point_angle % interval > 0.9 * interval:
                    point_angle += (interval - (point_angle % interval))
                # Append to respective id in sensor_data
                if int(np.round(point_angle / interval)) == 18:
                    car_data[0, int(car_counts[0])] = [point.point.x, point.point.y]
                    car_counts[0] += 1
                else:
                    car_data[int(np.round(point_angle / interval)), int(
                        car_counts[int(np.round(point_angle / interval))])] = [
                        point.point.x, point.point.y]
                    car_counts[int(np.round(point_angle / interval))] += 1
    car_data = np.sum(car_data, axis=-2)
    car_data = car_data / car_counts[:, None]
    car_data = np.linalg.norm(car_data, ord=2, axis=-1)
    car_data[np.argwhere(np.isnan(car_data))] = 40.
    car_data /= 40.0
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
                 steering_agent=None,
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
                                                           sun_altitude_angle=70.0))
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
        self.steering_agent = steering_agent

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

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2').to(gpt_device)
        self.gpt_model.eval()

    def resetPlayer(self):
        # Reset scenario_mode
        player_lane = -3  # random.choice([-2, -3])
        self.to_speed_up = True  # random.choice([True, False])
        self.speed_up_car = random.choice([0, 1])
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=player_lane, s=35.0)
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
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=0.0)
        other_car_waypoints.append(spawn_point)
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-3, s=0.0)
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
            if self.to_speed_up:
                if id == self.speed_up_car:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -300.)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -63.)
            else:
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -63.)
            self.traffic_manager.auto_lane_change(vehicle, False)
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
            id += 1
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                self.vehicles_list.append(response.actor_id)

        # traffic_manager.global_percentage_speed_difference(30.0)

    def attachCamera(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.image_resolution[0]))
        bp.set_attribute('image_size_y', str(self.image_resolution[1]))
        self.camera_rgb_front = self.world.spawn_actor(
            bp,
            # carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
            carla.Transform(carla.Location(x=0.2, y=-0.2, z=1.17), carla.Rotation(pitch=0.0)),
            # carla.Transform(carla.Location(z=1.45), carla.Rotation(pitch=0.0)),
            # carla.Transform(carla.Location(x=-0.7, y=0.0, z=1.4), carla.Rotation(yaw=180.0)),
            attach_to=self.player,
            attachment_type=carla.AttachmentType.Rigid)
        self.camera_rgb_back = self.world.spawn_actor(
            bp,
            # carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
            # carla.Transform(carla.Location(x=0.2, y=-0.2, z=1.17), carla.Rotation(pitch=0.0)),
            carla.Transform(carla.Location(x=-0.7, y=0.0, z=1.4), carla.Rotation(yaw=180.0)),
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

    def random_shooting(self, next_control=None):
        batch_size = 256
        time_length = 1
        frame_skip = 6
        action_mask_list = []
        for i in range(time_length):
            action_mask = []
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 0 1 2
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 3 4 5
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 10 11 12
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 13 14 15
            action_mask += [torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 3)]  # 16 17 18
            action_mask = torch.cat(action_mask, dim=-1)

            action_mask = [action_mask[..., 0:1], action_mask[:, :, 0:1], action_mask[:, :, 1:2],  # -180, -160, -140
                           action_mask[:, :, 1:2], action_mask[:, :, 1:2], action_mask[:, :, 2:3],  # -120, -100, -80
                           action_mask[:, :, 2:3], action_mask[:, :, 2:3], action_mask[:, :, 3:4],  # -60, -40, -20
                           action_mask[:, :, 3:4], action_mask[:, :, 3:4], action_mask[:, :, 4:5],  # 0, 20. 40
                           action_mask[:, :, 4:5], action_mask[:, :, 4:5], action_mask[:, :, 5:6],  # 60, 80, 100
                           action_mask[:, :, 5:6], action_mask[:, :, 5:6], action_mask[:, :, 0:1]]  # 120, 140, 160]

            action_mask = torch.cat(action_mask + [torch.ones(size=(frame_skip, batch_size, 5))], dim=-1)
            action_mask = torch.cat(
                [action_mask, torch.randint(low=0, high=2, size=(1, batch_size, 1)).repeat(frame_skip, 1, 1)], dim=-1)
            action_mask_list += [action_mask]
        action_mask = torch.cat(action_mask_list, dim=0)
        current_state = torch.tensor(self._state).unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(1, batch_size, 1)
        # print(current_state.shape)
        # print(action_mask)

        accumulated_q_value = self.prediction_mask(current_state, action_mask, next_control)
        # print(accumulated_q_value.size())
        max_id = torch.argmax(accumulated_q_value, dim=0)

        opti_action_mask = action_mask[:, max_id:max_id + 1]
        return opti_action_mask

    def prediction_mask(self, current_state: torch.Tensor, action_mask: torch.Tensor,
                        next_control: torch.Tensor = None):
        # current_state (1, batch, 2 * latent_size)
        # action_mask (time, batch, action (19+5))
        current_agent_state = current_state[..., :self.latent_size]
        current_env_state = current_state[..., self.latent_size:]

        time_length, batch_size, _ = action_mask.size()

        next_env_state = current_env_state
        next_agent_state = current_agent_state

        accumulated_q_value = 0.0
        discount_factor = 0.99
        for t in range(time_length):
            # given z_t, a_m_t, predict z_t+1
            # predict next env state
            next_env_state, agent_action = self.prediction(next_env_state, next_agent_state)

            # update human state
            # print(self._action_mask.shape)
            next_env_description = self.model.autoencoder.get_description_dist_normal(next_env_state).mean
            next_env_text_embeds = self.model.autoencoder.decoder_row_text(next_env_state)
            action_mask_t = action_mask[t:t + 1]

            # next_agent_state = self.model.autoencoder.get_latent_state_dist(
            #     description=next_env_description * action_mask_t,
            #     pre_state=next_agent_state,
            #     action=agent_action,
            #     mask=action_mask_t).sample()

            # mask the behind view.
            limited_view_mask = torch.ones_like(action_mask_t[..., :]).to(action_mask_t.device)
            limited_view_mask[..., 0:4] = 0.0
            limited_view_mask[..., -4 - 6:-6] = 0.0

            next_agent_state = self.model.autoencoder.get_latent_state_dist(
                description=next_env_description * action_mask_t[..., :-1] * limited_view_mask[..., :-1],
                text_embeds=next_env_text_embeds * action_mask_t[..., -1:] * limited_view_mask[..., -1:],
                pre_state=next_agent_state,
                action=agent_action,
                mask=action_mask_t * limited_view_mask).sample()

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

            mask_cost = torch.sum(action_mask_t[..., :-1], dim=-1, keepdim=True) * 0.15 \
                        + torch.sum(action_mask_t[..., -1:], dim=-1, keepdim=True) * 0.4
            # accumulated_q_value += discount_factor ** t * (self.get_q_value(next_env_state) - mask_cost)
            accumulated_q_value += discount_factor ** t * (self.get_q_value(next_env_state) - mask_cost)
        if next_control is not None:
            accumulated_q_value -= torch.abs(action_mask[0] - next_control.unsqueeze(dim=0)).sum(dim=-1,
                                                                                                 keepdim=True) * 0.0
        return accumulated_q_value.squeeze(dim=0)  # (batch, 1)

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
        text_embed_env, _ = generate_text_embed(env_state_torch, self.tokenizer, self.gpt_model)
        text_embed_env_torch = torch.tensor(text_embed_env).type(torch.float)
        self._latent_state_env = self.model.get_state_representation(observation=env_state_torch,
                                                                     text_embeds=text_embed_env_torch)

        env_description_agent_torch = torch.tensor(env_description_agent).type(torch.float)
        env_description_agent_torch[:18] = 1.0
        mask = torch.ones(size=env_description_agent_torch.size()).type(torch.float)
        self._action_mask = mask.detach().numpy().copy()

        text_embed_agent, _ = generate_text_embed(env_description_agent_torch, self.tokenizer, self.gpt_model)
        text_embed_agent_torch = torch.tensor(text_embed_agent).type(torch.float)
        self._latent_state_agent = self.model.get_state_representation(observation=env_description_agent_torch,
                                                                       text_embeds=text_embed_agent_torch)

        self._action, action_agent_dist = self.model.policy(self._latent_state_agent.unsqueeze(dim=0))
        self._action = torch.tanh(self._action[0])
        self._latent_state_agent = self._latent_state_agent.detach().numpy()
        self._latent_state_env = self._latent_state_env.detach().numpy()

        self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        return self._state.copy(), collided

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

    def get_env_description(self, env_state):
        return env_state.copy()

    def step_communication(self, action):
        rewards = []
        collideds = 0.0
        for _ in range(self.frame_skip):
            self._state, reward, collided, done, text_token, text_embed = self.forward(self._state, action)
            rewards.append(reward)
            if collided != 0.0:
                collideds = 1.0
        reward = np.mean(rewards)

        return self._state.copy(), reward, collideds, done, text_token, text_embed

    def forward(self, current_state: np.array, action_mask: np.array):
        # Update the latent state first, then sample action to forward environment
        # print(action_mask[1:7])
        self._env_state = self._env_state.copy()
        # self._env_state[:19] = 1.0
        self._env_description = self.get_env_description(self._env_state)

        text_embed, text_token = generate_text_embed(self._env_state, self.tokenizer, self.gpt_model)

        # latent_state_dist = self.model.autoencoder.get_latent_state_dist(
        #     description=description[t] * mask[t, ..., :-1],
        #     text_embeds=text_embeds[t] * mask[t, ..., -1:],
        #     pre_state=None,
        #     action=None,
        #     mask=mask[t])

        current_state_torch = torch.tensor(current_state).type(torch.float)
        self._action_mask = action_mask.copy()

        action_mask_torch = torch.tensor(self._action_mask).type(torch.float)  # .to('cpu')

        env_description_torch = torch.tensor(self._env_description).type(torch.float)
        text_embed_torch = torch.tensor(text_embed).type(torch.float)

        env_state_torch = torch.tensor(self._env_state).type(torch.float)
        latent_state_env_torch = torch.tensor(self._latent_state_env).type(torch.float)
        env_mask = torch.ones_like(action_mask_torch).to(action_mask_torch.device)
        env_mask[..., -1:] = 0
        self._latent_state_env = self.model.get_state_representation(observation=env_state_torch * env_mask[..., :-1],
                                                                     text_embeds=text_embed_torch * env_mask[..., -1:],
                                                                     pre_state=latent_state_env_torch,
                                                                     action=self._action,
                                                                     mask=env_mask)
        text_embed_torch = self.model.autoencoder.decoder_row_text(self._latent_state_env)
        predicted_text = generate_text(text_embed_torch, self.model.gpt_model_lm, self.tokenizer)

        limited_view_mask = torch.ones_like(action_mask_torch[..., :]).to(action_mask_torch.device)
        limited_view_mask[..., 0:4] = 0.0
        limited_view_mask[..., -4 - 6:-6] = 0.0

        self._latent_state_agent = self.model.autoencoder.get_latent_state_dist(
            description=env_description_torch * action_mask_torch[:-1] * limited_view_mask[..., :-1],
            text_embeds=text_embed_torch * action_mask_torch[-1:] * limited_view_mask[..., -1:],
            pre_state=current_state_torch[:self.latent_size],
            action=self._action,
            mask=action_mask_torch * limited_view_mask).mean
        # print(action_mask_torch * limited_view_mask)

        if (action_mask_torch * limited_view_mask)[...,-1].item() == 1.0:
            print(predicted_text)
        # print(limited_view_mask)

        # self._latent_state_agent = self.model.autoencoder.get_latent_state_dist(
        #     description=env_description_torch * action_mask_torch,
        #     pre_state=current_state_torch[:self.latent_size],
        #     action=self._action,
        #     mask=action_mask_torch).sample()

        # env_state_torch = torch.tensor(self._env_state).type(torch.float)
        # latent_state_env_torch = torch.tensor(self._latent_state_env).type(torch.float)
        # self._latent_state_env = self.model.get_state_representation(observation=env_state_torch,
        #                                                              pre_state=latent_state_env_torch,
        #                                                              action=self._action,
        #                                                              mask=None)
        # Physical action using the learnt basic model
        self._action, action_agent_dist = self.model.policy(self._latent_state_agent.unsqueeze(dim=0))
        self._action = action_agent_dist.mean
        self._action = torch.tanh(self._action[0])
        # Physical action using the steering wheel
        # self._action = self.steering_agent.parseVehicleWheel()
        # self._action = torch.tensor([self._action.throttle, self._action.steer])

        # print("env:", torch.tanh(self.model.policy(self._latent_state_env.unsqueeze(dim=0))[0]))
        # print("agent:", torch.tanh(self.model.policy(self._latent_state_agent.unsqueeze(dim=0))[0]))

        # self._action = self._action.detach().numpy().copy()

        self._latent_state_agent = self._latent_state_agent.detach().numpy()
        self._latent_state_env = self._latent_state_env.detach().numpy()
        self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        # add communication cost to reward
        # print(reward)
        for _ in range(3):
            self._env_state, reward, collided, done = self.simulator_step(self._action.detach().numpy().copy())
        # self._env_state, reward, collided, done = self.simulator_step(self._action.detach().numpy().copy())

        reward -= np.sum(self._action_mask[0:18]) * 0.25 * 1 / 3
        # print(np.sum(self._action_mask[0:19]) * 0.25 * 1 / 3)
        # print(action_mask_torch[:19].detach().numpy())
        return self._state, reward, collided, done, text_token, text_embed_torch.detach().cpu().numpy()

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
            self.player.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0., 0., 0.

        # Advance the simulation and wait for sensors data
        if self.render_display:
            snapshot, rgb_image_front, rgb_image_back, lidar_scan = self.sync_mode.tick(timeout=2.0)
        else:
            snapshot, lidar_scan = self.sync_mode.tick(timeout=2.0)
        # Render display
        if self.render_display:
            draw_image(self.display, rgb_image_front, rgb_image_back)
            # draw_front_image(self.display, rgb_image_front)
            self.hud.tick(self, self.clock)
            self.hud.render(self.display)
            pygame.display.flip()

        lidar_data, lidar_type = process_lidar(lidar_scan, self)
        if self._action_mask is not None:
            # self.visualize(lidar_data, lidar_type)
            pass
        next_obs = self.get_obs()
        next_obs = np.concatenate((lidar_data, next_obs), axis=0)
        reward, done, collided = self.getReward(steer, brake, lidar_data, lidar_type)

        if self.player.get_location().z > 1.:
            print("Episode done: vertical velocity too high, usually a simulator glitch")
            done = True

        return next_obs, reward, collided, done

    def getReward(self, steer, brake, lidar_scan, lidar_type):
        dist_from_center, vel_s, speed, done = self.dist_from_center_lane()
        vel_s_kmh = vel_s * 3.6
        collision_intensity = sum(self.collision_intensity)
        self.collision_intensity.clear()  # clear it ready for next time step
        assert collision_intensity >= 0.
        colliding = float(collision_intensity > 0.)
        if colliding:
            done, reward = True, -10.
        else:
            front_lidar_cost, surround_lidar_cost, back_lidar_cost = 0.0, 0.0, 0.0
            front_beam_index = int(lidar_scan.shape[0] / 2.)
            for i in range(lidar_scan.shape[0]):
                if i == front_beam_index and lidar_scan[i] < 0.5 and lidar_type[i] == 0:
                    front_lidar_cost = 2.
                if (i == front_beam_index - 1 or i == front_beam_index + 1) and \
                        lidar_scan[i] < 0.125 and lidar_type[i] == 0:
                    front_lidar_cost = 2.
                if lidar_scan[i] < 0.04 and lidar_type[i] == 0:
                    surround_lidar_cost = 4.
                if i == 0 and lidar_scan[i] < 0.5 and lidar_type[i] == 0:
                    back_lidar_cost = 2.
            wall_cost = 0.0
            # if dist_from_road_center > 0.7 or dist_from_road_center < -0.7:
            #     lidar_colliding_cost = 4.

            player_lane = self.map.get_waypoint(self.player.get_location()).lane_id
            lane_change_cost = 0. if player_lane == self.player_lane else 1.0
            lane_center_cost = dist_from_center / 1.75
            self.player_lane = player_lane

            speed_reward = vel_s_kmh / 40. if vel_s_kmh <= 40. else -(vel_s_kmh - 40.) / 40.

            reward = speed_reward - 0.3 * brake - 0.1 * abs(steer) - \
                     1.0 * (lane_change_cost + lane_center_cost) - \
                     front_lidar_cost - surround_lidar_cost - back_lidar_cost
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

    def visualize(self, lidar_data, lidar_type):
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
        pygame.display.flip()
