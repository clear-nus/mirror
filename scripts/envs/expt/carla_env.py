import carla
import pygame
import cv2

import random
import numpy as np
import torch
import math
import queue
import time
import os

from models.text_model import AutoEncodersUni

from envs.bounding_box import get_bounding_boxes, draw_bounding_boxes

from utils.audio_player import play, stop

from pytorch_transformers import GPT2Tokenizer


def seperate_lidar_text_mask(action_mask):
    lidar_action_mask = action_mask[..., :3 * 36 + 5]
    text_action_mask = action_mask[..., 3 * 36 + 5:]
    return lidar_action_mask, text_action_mask


def generate_text(lidar, speed_value, h_dir, v_dir, detect_threshold=0.8):
    if lidar < detect_threshold:
        if v_dir == 'rear':
            if speed_value.item() == 1:
                text = f"Car is approaching fast from your {h_dir}"
            else:
                text = f"Car is moving slowly at your {h_dir}"
        else:
            if speed_value.item() == 1:
                text = f"Car is moving fast at your {h_dir}"
            else:
                text = f"Car is slowing down at your {h_dir}"
    else:
        text = f"No car detected at your {h_dir}"

    # print(f'({v_dir},{h_dir}, {lidar.item()}): {text}')
    return text


def generate_text_np(obs, speed, device="cuda:1"):
    speed_threshold = 32
    detect_threshold = 0.8
    # generate h_dir == 'front'
    front_lidar = torch.tensor([obs[3 * 18 + 1]]).to(torch.float)
    front_speed = torch.tensor([0]).to(torch.float)
    if front_lidar < detect_threshold:
        front_speed = torch.tensor([speed[18]])
    h_dir = 'front'
    v_dir = 'front'
    if front_speed > speed_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)

    front_text = generate_text(front_lidar, speed_value, h_dir, v_dir, detect_threshold)

    # generate h_dir == 'left-front'
    left_front_lidar = torch.tensor([1]).to(torch.float)
    left_front_speed = torch.tensor([0]).to(torch.float)
    for i in range(8):
        if torch.tensor([obs[3 * (-i + 17) + 1]]).to(torch.float) < 0.8:
            left_front_lidar = torch.tensor([obs[3 * (-i + 17) + 1]]).to(torch.float)
            left_front_speed = speed[-i + 17]
            break
    h_dir = 'left-front'
    v_dir = 'front'
    if left_front_speed > speed_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)

    left_front_text = generate_text(left_front_lidar, speed_value, h_dir, v_dir, detect_threshold)

    # generate h_dir == 'right-front'
    right_front_lidar = torch.tensor([1]).to(torch.float)
    right_front_speed = torch.tensor([0]).to(torch.float)
    for i in range(8):
        if torch.tensor([obs[3 * (i + 19) + 1]]).to(torch.float) < detect_threshold:
            right_front_lidar = torch.tensor([obs[3 * (i + 19) + 1]]).to(torch.float)
            right_front_speed = speed[i + 19]
            break

    h_dir = 'right-front'
    v_dir = 'front'
    if right_front_speed > speed_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)

    right_front_text = generate_text(right_front_lidar, speed_value, h_dir, v_dir, detect_threshold)

    # generate h_dir == 'rear'
    rear_lidar = torch.tensor([obs[3 * 0 + 1]]).to(torch.float)
    rear_speed = torch.tensor([0]).to(torch.float)
    if rear_lidar < detect_threshold:
        rear_speed = torch.tensor([speed[0]]).to(torch.float)
    h_dir = 'rear'
    v_dir = 'rear'
    if rear_speed > speed_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)

    rear_text = generate_text(rear_lidar, speed_value, h_dir, v_dir, detect_threshold)

    # generate h_dir == 'left_rear'
    left_rear_lidar = torch.tensor([1]).to(torch.float)
    left_rear_speed = torch.tensor([0]).to(torch.float)
    for i in range(8):
        if torch.tensor([obs[3 * (i + 1) + 1]]).to(torch.float) < detect_threshold:
            left_rear_lidar = torch.tensor([obs[3 * (i + 1) + 1]]).to(torch.float)
            left_rear_speed = speed[i + 1]
            break
    h_dir = 'left-rear'
    v_dir = 'rear'
    if left_rear_speed > speed_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)

    left_rear_text = generate_text(left_rear_lidar, speed_value, h_dir, v_dir, detect_threshold)

    # generate h_dir == 'right-rear'
    right_rear_lidar = torch.tensor([1]).to(torch.float)
    right_rear_speed = torch.tensor([0]).to(torch.float)
    for i in range(8):
        if torch.tensor([obs[3 * (-i + 35) + 1]]).to(torch.float) < detect_threshold:
            right_rear_lidar = torch.tensor([obs[3 * (-i + 35) + 1]]).to(torch.float)
            right_rear_speed = speed[-i + 35]
            break
    h_dir = 'right-rear'
    v_dir = 'rear'
    if right_rear_speed > speed_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)

    right_rear_text = generate_text(right_rear_lidar, speed_value, h_dir, v_dir, detect_threshold)

    text_np = np.array([front_text, left_front_text, right_front_text, rear_text, left_rear_text, right_rear_text])
    return text_np


def generate_text_embed(text_model, obs: torch.Tensor, speed, device="cuda:1"):
    spped_threshold = 32
    detect_threshold = 0.8
    # generate h_dir == 'front'
    front_lidar = torch.tensor([obs[3 * 18 + 1]]).to(torch.float).to(device)
    front_speed = torch.tensor([0]).to(torch.float).to(device)
    if front_lidar < detect_threshold:
        front_speed = torch.tensor([speed[18]]).to(device)
    h_dir = 'front'
    v_dir = 'front'
    if front_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float).to(device)
    else:
        speed_value = torch.tensor([0]).to(torch.float).to(device)
    front_text_embed = text_model.get_latent_dist(front_lidar, speed_value, h_dir,
                                                  v_dir).mean.detach().cpu().numpy() / 15.0
    front_text_embed = np.concatenate([front_text_embed, front_lidar.cpu().detach().numpy()], axis=-1)
    # if front_lidar < detect_threshold:
    #     print(f'{h_dir}, {speed_value}, {front_speed}')
    # else:
    #     print(f'no cat detected {h_dir}')

    # generate h_dir == 'left-front'
    left_front_lidar = torch.tensor([1]).to(torch.float).to(device)
    left_front_speed = torch.tensor([0]).to(torch.float).to(device)
    for i in range(8):
        if torch.tensor([obs[3 * (-i + 17) + 1]]).to(torch.float) < 0.8:
            left_front_lidar = torch.tensor([obs[3 * (-i + 17) + 1]]).to(torch.float).to(device)
            left_front_speed = speed[-i + 17]
            break
    h_dir = 'left-front'
    v_dir = 'front'
    if left_front_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float).to(device)
    else:
        speed_value = torch.tensor([0]).to(torch.float).to(device)
    left_front_text_embed = text_model.get_latent_dist(left_front_lidar, speed_value, h_dir,
                                                       v_dir).mean.detach().cpu().numpy() / 15.0
    left_front_text_embed = np.concatenate([left_front_text_embed, left_front_lidar.cpu().detach().numpy()], axis=-1)
    # if left_front_lidar < detect_threshold:
    #     print(f'{h_dir}, {speed_value}, {left_front_speed}')
    # else:
    #     print(f'no cat detected {h_dir}')

    # generate h_dir == 'right-front'
    right_front_lidar = torch.tensor([1]).to(torch.float).to(device)
    right_front_speed = torch.tensor([0]).to(torch.float).to(device)
    for i in range(8):
        if torch.tensor([obs[3 * (i + 19) + 1]]).to(torch.float) < detect_threshold:
            right_front_lidar = torch.tensor([obs[3 * (i + 19) + 1]]).to(torch.float).to(device)
            right_front_speed = speed[i + 19]
            break

    h_dir = 'right-front'
    v_dir = 'front'
    if right_front_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float).to(device)
    else:
        speed_value = torch.tensor([0]).to(torch.float).to(device)
    right_front_text_embed = text_model.get_latent_dist(right_front_lidar, speed_value,
                                                        h_dir, v_dir).mean.detach().cpu().numpy()
    right_front_text_embed = np.concatenate([right_front_text_embed, right_front_lidar.cpu().detach().numpy()],
                                            axis=-1) / 15.0
    # if right_front_lidar < detect_threshold:
    #     print(f'{h_dir}, {speed_value}, {right_front_speed}')
    # else:
    #     print(f'no cat detected {h_dir}')

    # generate h_dir == 'rear'
    rear_lidar = torch.tensor([obs[3 * 0 + 1]]).to(torch.float).to(device)
    rear_speed = torch.tensor([0]).to(torch.float).to(device)
    if rear_lidar < detect_threshold:
        rear_speed = torch.tensor([speed[0]]).to(torch.float).to(device)
    h_dir = 'rear'
    v_dir = 'rear'
    if rear_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float).to(device)
    else:
        speed_value = torch.tensor([0]).to(torch.float).to(device)
    rear_text_embed = text_model.get_latent_dist(rear_lidar, speed_value, h_dir,
                                                 v_dir).mean.detach().cpu().numpy() / 15.0
    rear_text_embed = np.concatenate([rear_text_embed, rear_lidar.cpu().detach().numpy()], axis=-1)

    # if rear_lidar < detect_threshold:
    #     print(f'{h_dir}, {speed_value}, {speed[0]}')
    # else:
    #     print(f'no cat detected {h_dir}')

    # generate h_dir == 'left_rear'
    left_rear_lidar = torch.tensor([1]).to(torch.float).to(device)
    left_rear_speed = torch.tensor([0]).to(torch.float).to(device)
    for i in range(8):
        if torch.tensor([obs[3 * (i + 1) + 1]]).to(torch.float) < detect_threshold:
            left_rear_lidar = torch.tensor([obs[3 * (i + 1) + 1]]).to(torch.float).to(device)
            left_rear_speed = speed[i + 1]
            break
    h_dir = 'left-rear'
    v_dir = 'rear'
    if left_rear_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float).to(device)
    else:
        speed_value = torch.tensor([0]).to(torch.float).to(device)
    left_rear_text_embed = text_model.get_latent_dist(left_rear_lidar, speed_value, h_dir,
                                                      v_dir).mean.detach().cpu().numpy() / 15.0
    left_rear_text_embed = np.concatenate([left_rear_text_embed, left_rear_lidar.cpu().detach().numpy()], axis=-1)

    # if left_rear_lidar < detect_threshold:
    #     print(f'{h_dir}, {speed_value}, {left_rear_speed}')
    # else:
    #     print(f'no cat detected {h_dir}')

    # generate h_dir == 'right-rear'
    right_rear_lidar = torch.tensor([1]).to(torch.float).to(device)
    right_rear_speed = torch.tensor([0]).to(torch.float).to(device)
    for i in range(8):
        if torch.tensor([obs[3 * (-i + 35) + 1]]).to(torch.float) < detect_threshold:
            right_rear_lidar = torch.tensor([obs[3 * (-i + 35) + 1]]).to(torch.float).to(device)
            right_rear_speed = speed[-i + 35]
            break
    h_dir = 'right-rear'
    v_dir = 'rear'
    if right_rear_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float).to(device)
    else:
        speed_value = torch.tensor([0]).to(torch.float).to(device)
    right_rear_text_embed = text_model.get_latent_dist(right_rear_lidar, speed_value, h_dir,
                                                       v_dir).mean.detach().cpu().numpy() / 15.0
    right_rear_text_embed = np.concatenate([right_rear_text_embed, right_rear_lidar.cpu().detach().numpy()], axis=-1)

    # if right_rear_lidar < detect_threshold:
    #     print(f'{h_dir}, {speed_value}, {right_rear_speed}')
    # else:
    #     print(f'no cat detected {h_dir}')

    text_embed = np.concatenate([front_text_embed, left_front_text_embed, right_front_text_embed,
                                 rear_text_embed, left_rear_text_embed, right_rear_text_embed], axis=-1)
    # print(text_embed)
    # return np.zeros(20 * 6)
    return text_embed


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
    front_array = front_array[int(image_front.height / 4):int(3 * image_front.height / 4), :, :]
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
                        car_data_dheight[0, int(car_dheight_counts[0, 0]), 0] = [point.point.x, point.point.y,
                                                                                 point.point.z]
                        car_dheight_counts[0, 0] += 1
                    elif -1.2 <= point.point.z < -0.7:
                        car_data_dheight[0, int(car_dheight_counts[0, 1]), 1] = [point.point.x, point.point.y,
                                                                                 point.point.z]
                        car_dheight_counts[0, 1] += 1
                    elif point.point.z >= -0.7:
                        car_data_dheight[0, int(car_dheight_counts[0, 2]), 2] = [point.point.x, point.point.y,
                                                                                 point.point.z]
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
                    if -1.2 <= point.point.z < -0.7:
                        car_data_dheight[int(np.round(point_angle / interval)), int(
                            car_dheight_counts[int(np.round(point_angle / interval)), 1]), 1] = [point.point.x,
                                                                                                 point.point.y,
                                                                                                 point.point.z]
                        car_dheight_counts[int(np.round(point_angle / interval)), 1] += 1
                    elif point.point.z >= -0.0:
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
                                             np.expand_dims(
                                                 np.count_nonzero(car_data_id_points == env.vehicles_list[i], axis=-1),
                                                 axis=-1)),
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


class CarlaEnv():
    def __init__(self,
                 render_display=True,
                 render_fog=True,
                 expt_stage='warmup-clear',
                 host="127.0.0.1",
                 port=2000,
                 tm_port=8000,
                 frame_skip=1,
                 image_resolution=(1280, 720),
                 robot_model=None,
                 mirror_model=None,
                 steering_agent=None,
                 forward_record_flag=False):
        self.render_display = render_display
        self.ready_mode = True
        self.survey_mode = False
        self.expt_stage = expt_stage  # 'warmup-clear', 'warmup-fog', 'expt'
        self.prev_text = ""
        self.speech_time = time.time()
        self.episode = 1
        self.total_episodes = 3
        self.image_resolution = image_resolution
        self.frame_skip = frame_skip
        self.actor_list = []
        # Setup display
        if self.render_display:
            pygame.init()
            self.display = pygame.display.set_mode(
                self.image_resolution,
                pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.SCALED)
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
        self.render_fog = render_fog
        self.setWeather()
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
        # self.goal_left_wpt = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=380.0)
        self.goal_left_wpt = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=360.0)
        self.goal_right_wpt = self.goal_left_wpt.get_right_lane()
        self.goal_left_loc = self.goal_left_wpt.transform.location
        self.goal_right_loc = self.goal_right_wpt.transform.location
        self.all_goal_lanes_loc, wpt = [], self.goal_left_wpt
        for i in range(2):
            self.all_goal_lanes_loc.append(wpt.transform.location)
            wpt = wpt.get_right_lane()
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

        self.model_device = 'cuda:0'
        self.condition = 'noop'  # noop: no communication; mirror: self projection

        self.robot_model = robot_model.to(self.model_device)
        self.mirror_model = mirror_model.to(self.model_device)

        self.text_model = AutoEncodersUni(latent_size=19, hidden_size=64, layers=3)
        self.text_model = torch.load('./current_model/autoencoder.pt')
        self.text_model.set_device(self.model_device)

        self.latent_size = self.robot_model.latent_size

        self.lidar_action_mask_list = None
        self.text_action_mask_list = None

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.gpt_lm_model = []
        for i in range(6):
            self.gpt_lm_model += [torch.load(f'./current_model/gpt_lm_model_{i}.pt').to(self.model_device)]

        self._action_mask = None
        self.lidar_p = None
        self.text_p = None

        # if record human demonstration data
        self.record_demonstration_flag = True
        self.forward_record_flag = forward_record_flag
        self.demonstration_save_path = '.'

        capacity = 15000
        self.latent_state_buffer = np.empty(shape=(capacity, 128))
        self.lidar_obs_buffer = np.empty(shape=(capacity, 36 * 3 + 5))
        self.text_embed_buffer = np.empty(shape=(capacity, 20 * 6))
        self.human_actions_buffer = np.empty(shape=(capacity, 2))
        self.demonstration_count = 0
        self.pre_q_value = 0
        # perceptual mask for human
        self.perceptual_mask = None

        self.exp_case = None

        self.front_mask = np.zeros((600, 800, 1))
        self.back_mask = np.zeros((55, 360, 1))

        for i in range(600):
            if i < 110:
                self.front_mask[i][0:int(165 + 0.2 * i)] = 1
                self.front_mask[i][int(635 - 0.2 * i):] = 1
            if 110 <= i < 200:
                self.front_mask[i][:] = 1
            if 200 <= i < 300:
                self.front_mask[i][:] = 1
            if 300 <= i < 335:
                self.front_mask[i][int(1.9 * (i - 300)):int(800 - 1.9 * (i - 300))] = 1
            if 335 <= i < 350:
                self.front_mask[i][int(1.9 * (i - 300)):int(190 - 1.2 * (i - 300))] = 1
                self.front_mask[i][int(190 + 1.2 * (i - 300)):int(800 - 1.9 * (i - 300))] = 1

        for i in range(23):
            self.back_mask[i][int(105 + 1.5 * i): int(255 - 1.5 * i)] = 1

        self.blank_front = np.ones((600, 800, 3)) * 190
        self.blank_back = np.ones((55, 360, 3)) * 200

        self.front_blur_mask = np.ones((600, 800, 1))
        self.back_blur_mask = np.ones((55, 360, 1))

        for i in range(350):
            if i < 323:
                self.front_blur_mask[i, :] = (1.0 - (i / 7000.0))
            else:
                self.front_blur_mask[i, :] = (1.0 - (i / 7000.0) - (i - 323) / 30)

        for i in range(400):
            if i < 200:
                self.front_blur_mask[:, 400 + i] *= (1.0 - (i / 7000.0))
                self.front_blur_mask[:, 400 - i] *= (1.0 - (i / 7000.0))
            else:
                self.front_blur_mask[:, 400 + i] *= ((1.0 - (i / 7000.0)) * (1.0 - ((i - 200) / 200.0)))
                self.front_blur_mask[:, 400 - i] *= ((1.0 - (i / 7000.0)) * (1.0 - ((i - 200) / 200.0)))
        self.front_blur_mask = self.front_blur_mask.clip(min=0, max=255)

        for i in range(55):
            self.back_blur_mask[i, :] = (1.0 - (i / 105.0))
        self.back_blur_mask = self.back_blur_mask.clip(min=0, max=255)

    def set_perceptual_mask(self, perceptual_mask):
        self.perceptual_mask = perceptual_mask.to(self.model_device)

    def set_policy_filter(self, policy_filter):
        self.policy_filter = policy_filter.to(self.model_device)

    def setWeather(self):
        if self.render_fog:
            self.world.set_weather(carla.WeatherParameters(cloudiness=100.0,
                                                           precipitation=0.0,
                                                           fog_density=100.0,
                                                           sun_altitude_angle=70.0,
                                                           fog_distance=0.0))

    def resetPlayer(self, ego_car=None, other_car=None):
        # Reset scenario_mode
        if ego_car is None:
            player_lane = random.choice([-3])
            self.speed_up_down = random.choice([2, 4])  # no-op/up-left/up-right/down-left/down-right

        if ego_car == 'l':
            player_lane = -2
        elif ego_car == 'r':
            player_lane = -3

        if other_car == 'lf':
            self.speed_up_down = 3
            print('scenario: down-left')
        elif other_car == 'lr':
            self.speed_up_down = 1
            print('scenario: up-left')
        if other_car == 'rf':
            self.speed_up_down = 4
            print('scenario: down-right')
        elif other_car == 'rr':
            self.speed_up_down = 2
            print('scenario: up-right')

        self.exp_case = f'{ego_car}-{other_car}'

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

    def resetOtherVehiclesWarmup(self):
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
        left_waypoint = self.map.get_waypoint(self.player.get_location())
        if left_waypoint.lane_id != -2:
            left_waypoint = left_waypoint.get_left_lane()
        num_vehicles = 5
        distance = 40.
        other_car_waypoints = []
        for _ in range(num_vehicles):
            left_waypoint = left_waypoint.next(distance)[0]
            waypoint = left_waypoint
            # Randomize lane to spawn
            lane_id = -random.randint(2, 3)
            if lane_id == -3:
                waypoint = waypoint.get_right_lane()
            other_car_waypoints.append(waypoint)
            distance = float(random.randint(40, 70))

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
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 100.)
            self.traffic_manager.auto_lane_change(vehicle, False)
            id += 1
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                self.vehicles_list.append(response.actor_id)

        # traffic_manager.global_percentage_speed_difference(30.0)

    def resetOtherVehicles(self, speed_up_speed=None, rear_spawn_pos=None, front_spawn_pos=None, slow_down_speed=None):
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
        # Randomize all speed up/slow down
        if speed_up_speed is None:
            self.speed_up_speed = random.randint(120, 130)
        else:
            self.speed_up_speed = speed_up_speed

        if rear_spawn_pos is None:
            self.rear_spawn_pos = float(random.randint(0, 10))
        else:
            self.rear_spawn_pos = rear_spawn_pos
            # self.slow_down_pos = float(random.randint(230, 300))
        self.slow_down_pos = 190.0 # float(random.randint(180, 200))

        if front_spawn_pos is None:
            self.front_spawn_pos = float(random.randint(115, 125))
        else:
            self.front_spawn_pos = front_spawn_pos

        if slow_down_speed is None:
            self.slow_down_speed = float(random.randint(80, 90))
        else:
            self.slow_down_speed = slow_down_speed
        # Rear two cars
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=self.rear_spawn_pos)
        other_car_waypoints.append(spawn_point)
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-3, s=self.rear_spawn_pos)
        other_car_waypoints.append(spawn_point)
        # Front two cars
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-2, s=self.front_spawn_pos)
        other_car_waypoints.append(spawn_point)
        spawn_point = self.map.get_waypoint_xodr(road_id=22, lane_id=-3, s=self.front_spawn_pos)
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
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 100.)
            id += 1
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                self.vehicles_list.append(response.actor_id)

        # traffic_manager.global_percentage_speed_difference(30.0)

    def accelOtherVehicles(self):
        for count, actor_id in enumerate(self.vehicles_list):
            vehicle = self.world.get_actor(actor_id)
            if self.speed_up_down == 1:  # up-left
                if count == 0 or count == 2:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -float(self.speed_up_speed))
                elif count == 1:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -35.0)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -64.0)
            elif self.speed_up_down == 2:  # up-right
                if count == 1 or count == 3:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -float(self.speed_up_speed))
                elif count == 0:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -35.0)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -64.0)
            else:  # no-op
                if count == 0 or count == 1:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -35.0)
                else:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -64.0)
            self.traffic_manager.auto_lane_change(vehicle, False)
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)

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
            carla.Transform(carla.Location(x=0.3, y=0.0, z=1.29), carla.Rotation(pitch=-4.0, yaw=180.0)),
            # camera at the rear minor
            # carla.Transform(carla.Location(x=-1.0, y=0.0, z=1.4), carla.Rotation(yaw=180.0)),
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

    def adaptive_random_shooting(self):
        with torch.no_grad():
            sample_size = 1000
            max_itr = 3
            time_length = 4
            frame_skip = 2

            text_p_tau = 0.3
            lidar_p_tau = 0.15

            mix_rate = 0.75
            lidar_p = np.ones(shape=(time_length * frame_skip, 6)) * 0.35
            text_p = np.ones(shape=(time_length * frame_skip, 6)) * 0.35
            if self.lidar_p is not None:
                lidar_p[:-1] = (1 - mix_rate) * lidar_p[:-1] + mix_rate * self.lidar_p[1:]
            if self.text_p is not None:
                text_p[:-1] = (1 - mix_rate) * text_p[:-1] + mix_rate * self.text_p[1:]

            update_rate = 0.8

            for i in range(max_itr):
                _, _, _, accumulated_q_value, lidar_action_mask, text_action_mask = self.random_shooting(lidar_p,
                                                                                                         text_p,
                                                                                                         sample_size=sample_size,
                                                                                                         time_length=time_length,
                                                                                                         frame_skip=frame_skip)
                top_k_ids = torch.topk(accumulated_q_value, dim=0, k=int(200)).indices.squeeze(dim=-1).to(
                    self.model_device)
                lidar_action_mask = torch.index_select(lidar_action_mask, dim=1, index=top_k_ids).detach().cpu().numpy()
                text_action_mask = torch.index_select(text_action_mask, dim=1, index=top_k_ids).detach().cpu().numpy()

                lidar_action_mask = np.concatenate([lidar_action_mask[..., 3 * 1 - 1:3 * 1],
                                                    lidar_action_mask[..., 3 * 1 + 3 * 8 - 1:3 * 1 + 3 * 8],
                                                    lidar_action_mask[...,
                                                    3 * 1 + 3 * 8 + 3 * 9 - 1:3 * 1 + 3 * 8 + 3 * 9],
                                                    lidar_action_mask[...,
                                                    3 * 1 + 3 * 8 + 3 * 9 + 3 * 1 - 1:3 * 1 + 3 * 8 + 3 * 9 + 3 * 1],
                                                    lidar_action_mask[...,
                                                    3 * 1 + 3 * 8 + 3 * 9 + 3 * 1 + 3 * 9 - 1:3 * 1 + 3 * 8 + 3 * 9 + 3 * 1 + 3 * 9],
                                                    lidar_action_mask[...,
                                                    3 * 1 + 3 * 8 + 3 * 9 + 3 * 1 + 3 * 9 + 3 * 8 - 1:3 * 1 + 3 * 8 + 3 * 9 + 3 * 1 + 3 * 9 + 3 * 8]],
                                                   axis=-1)
                text_action_mask = np.concatenate([text_action_mask[..., 20 * 1 - 1:20 * 1],
                                                   text_action_mask[..., 20 * 2 - 1:20 * 2],
                                                   text_action_mask[..., 20 * 3 - 1:20 * 3],
                                                   text_action_mask[..., 20 * 4 - 1:20 * 4],
                                                   text_action_mask[..., 20 * 5 - 1:20 * 5],
                                                   text_action_mask[..., 20 * 6 - 1:20 * 6]],
                                                  axis=-1)

                update_lidar_p = lidar_action_mask.mean(axis=1)
                update_text_p = text_action_mask.mean(axis=1)

                lidar_p = (1 - update_rate) * lidar_p + update_rate * update_lidar_p
                text_p = (1 - update_rate) * text_p + update_rate * update_text_p

            self.lidar_p = lidar_p.copy()
            self.text_p = text_p.copy()
            # print(max_id)
            lidar_p_torch = torch.as_tensor(lidar_p).to(self.model_device)
            text_p_torch = torch.as_tensor(text_p).to(self.model_device)

            opti_lidar_action_mask = (lidar_p_torch[0] > lidar_p_tau).type(torch.float)
            opti_text_action_mask = (text_p_torch[0] > text_p_tau).type(torch.float)

            opti_lidar_action_mask = [opti_lidar_action_mask[0:1].repeat(3 * 1),
                                      opti_lidar_action_mask[1:2].repeat(3 * 8),  # 0 1 2
                                      opti_lidar_action_mask[2:3].repeat(3 * 9),  # 3 4 5
                                      opti_lidar_action_mask[3:4].repeat(3 * 1),  # 6 7 8
                                      opti_lidar_action_mask[4:5].repeat(3 * 9),  # 9
                                      opti_lidar_action_mask[5:6].repeat(3 * 8),
                                      torch.ones(size=(5,)).to(self.model_device)]

            opti_text_action_mask = [opti_text_action_mask[0:1].repeat(20),
                                     opti_text_action_mask[1:2].repeat(20),  # 0 1 2
                                     opti_text_action_mask[2:3].repeat(20),  # 3 4 5
                                     opti_text_action_mask[3:4].repeat(20),  # 6 7 8
                                     opti_text_action_mask[4:5].repeat(20),  # 9
                                     opti_text_action_mask[5:6].repeat(20)]

            opti_lidar_action_mask = torch.cat(opti_lidar_action_mask, dim=-1)
            opti_text_action_mask = torch.cat(opti_text_action_mask, dim=-1)

            opti_action_mask = torch.cat([opti_lidar_action_mask, opti_text_action_mask], dim=-1)
            return opti_action_mask, opti_lidar_action_mask, opti_text_action_mask, \
                   (lidar_p.copy()[0] > lidar_p_tau).astype(np.float), (text_p.copy()[0] > text_p_tau).astype(np.float)

    def random_shooting(self, lidar_p=None, text_p=None, sample_size=12000, time_length=3, frame_skip=2):
        sample_size = sample_size
        time_length = time_length
        frame_skip = frame_skip

        if lidar_p is None:
            lidar_p = np.ones(shape=(time_length, 6)) * 0.5

        if text_p is None:
            text_p = np.ones(shape=(time_length, 6)) * 0.5
        # if self.lidar_action_mask_list is None:
        self.lidar_action_mask_list = []
        for i in range(time_length):
            self.lidar_action_mask = []
            temp_front = (torch.rand(size=(1, sample_size, 1)) < lidar_p[i, 0].item()).type(torch.float).repeat(frame_skip, 1, 3 * 2)
            self.lidar_action_mask += [temp_front]
            self.lidar_action_mask += [(torch.rand(size=(1, sample_size, 1)) < lidar_p[i, 1].item()).type(torch.float).repeat(frame_skip, 1, 3 * 7)]
            self.lidar_action_mask += [(torch.rand(size=(1, sample_size, 1)) < lidar_p[i, 2].item()).type(torch.float).repeat(frame_skip, 1, 3 * 8)]
            self.lidar_action_mask += [(torch.rand(size=(1, sample_size, 1)) < lidar_p[i, 3].item()).type(torch.float).repeat(frame_skip, 1, 3 * 3)]
            self.lidar_action_mask += [(torch.rand(size=(1, sample_size, 1)) < lidar_p[i, 4].item()).type(torch.float).repeat(frame_skip, 1, 3 * 8)]
            self.lidar_action_mask += [(torch.rand(size=(1, sample_size, 1)) < lidar_p[i, 5].item()).type(torch.float).repeat(frame_skip, 1, 3 * 7)]
            self.lidar_action_mask += [temp_front[..., 0:3]]

            self.lidar_action_mask = torch.cat(
                self.lidar_action_mask + [torch.ones(size=(frame_skip, sample_size, 5))], dim=-1)
            self.lidar_action_mask_list += [self.lidar_action_mask]
        self.lidar_action_mask_list = torch.cat(self.lidar_action_mask_list, dim=0).to(self.model_device)

        # if self.text_action_mask_list is None:
        self.text_action_mask_list = []
        for i in range(time_length):
            self.text_action_mask = []
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 0].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                      20)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 1].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                      20)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 2].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                      20)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 3].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                      20)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 4].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                      20)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 5].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                      20)]

            self.text_action_mask = torch.cat(self.text_action_mask, dim=-1)
            self.text_action_mask_list += [self.text_action_mask]
        self.text_action_mask_list = torch.cat(self.text_action_mask_list, dim=0).to(self.model_device)

        current_state = torch.tensor(self._state).unsqueeze(dim=0).unsqueeze(dim=0).to(self.model_device)
        current_state = current_state.repeat(1, sample_size, 1)

        lidar_action_mask = self.lidar_action_mask_list
        text_action_mask = self.text_action_mask_list
        accumulated_q_value = self.prediction_mask(current_state, lidar_action_mask, text_action_mask)
        max_id = torch.argmax(accumulated_q_value, dim=0).item()
        opti_lidar_action_mask = self.lidar_action_mask_list[0, max_id]
        opti_text_action_mask = self.text_action_mask_list[0, max_id]
        opti_action_mask = torch.cat([opti_lidar_action_mask, opti_text_action_mask], dim=-1)
        return opti_action_mask, opti_lidar_action_mask, opti_text_action_mask, \
               accumulated_q_value, self.lidar_action_mask_list, self.text_action_mask_list

    def prediction_mask(self, current_state, lidar_action_mask, text_action_mask):
        # current_state (1, batch, 2 * latent_size)
        # action_mask (time, batch, action (19+5))
        current_agent_state = current_state[..., :self.latent_size]
        current_env_state = current_state[..., self.latent_size:]

        time_length, batch_size, _ = lidar_action_mask.size()

        next_env_state = current_env_state
        next_agent_state = current_agent_state

        accumulated_q_value = 0.0
        discount_factor = 0.9

        human_model = self.mirror_model

        decoder_lidar = self.robot_model.autoencoder.decoder_lidar
        decoder_text = self.robot_model.autoencoder.decoder_text
        for t in range(time_length):
            # given z_t, a_m_t, predict z_t+1
            # predict next env state
            next_env_state, agent_action = self.prediction(human_model, next_env_state, next_agent_state)

            # update human state
            next_lidar_obs = decoder_lidar(next_env_state).mean
            next_text_obs = decoder_text(next_env_state).mean

            limited_view_mask = torch.ones_like(lidar_action_mask[t:t + 1]).to(lidar_action_mask.device)
            limited_view_mask[..., 0:8 * 3] = 0.0
            limited_view_mask[..., 28 * 3:36 * 3] = 0.0


            lidar_obs_2d = next_lidar_obs[..., :-5][..., 1::3]
            lidar_obs_mask = self.perceptual_mask.get_obs_mask(lidar_obs_2d)
            lidar_obs_mask_rsample_list = [lidar_obs_mask[:, :, k:k + 1].repeat(1, 1, 3) for k in range(36)]

            lidar_obs_mask = torch.cat(lidar_obs_mask_rsample_list +
                                           [torch.ones_like(lidar_obs_mask[:, :, 0:0 + 1]).repeat(1, 1, 5).to(
                                               self.model_device)], dim=-1) #* 0.0

            lidar_mask = (lidar_obs_mask + lidar_action_mask[t:t + 1] * limited_view_mask > 0.5).type(torch.float)

            next_agent_state = human_model.get_state_representation(next_lidar_obs,
                                                                    next_text_obs,
                                                                    action=agent_action,
                                                                    pre_state=next_agent_state,
                                                                    lidar_mask=lidar_mask,
                                                                    text_mask=text_action_mask[t:t + 1])

            # get Q value
            mask_cost = (torch.sum(lidar_action_mask[t:t + 1], dim=-1, keepdim=True) / 3.0) * 0.02 \
                        + (torch.sum(text_action_mask[t:t + 1], dim=-1, keepdim=True) / 20.0).pow(3) * 0.4
            accumulated_q_value += discount_factor ** t * (- mask_cost)

            accumulated_q_value += discount_factor ** t * (
                        self.robot_model.reward_model(next_env_state).mean * 10.0 - mask_cost)

        return accumulated_q_value.squeeze(dim=0)  # (batch, 1)

    def prediction(self, human_model, current_env_state, current_agent_state):
        # human take an action based current agent state
        action, action_agent_dist = human_model.policy(current_agent_state)
        residule_action = self.policy_filter(current_agent_state).mean
        action = torch.tanh(action+0.1 * residule_action)

        # forward environment based on current env state
        next_state_dist = self.robot_model.autoencoder.transition(torch.cat([current_env_state, action], dim=-1))
        return next_state_dist.mean, action

    def get_current_env_state(self):
        return torch.tensor(self._latent_state_env)

    def get_current_agent_state(self):
        return torch.tensor(self._latent_state_agent)

    def resetWarmup(self):
        self.resetPlayer()
        self.world.tick()
        self.resetOtherVehiclesWarmup()
        self.world.tick()
        self.collision_intensity = []
        self.init_accel = False
        self.prev_text = ""
        obs, lidar_data_id, lidar_data_vel, reward, collided, done = self.simulator_step(action=None, frame_count=0)
        text_embed = generate_text_embed(self.text_model, obs, lidar_data_vel, self.model_device)

        self._lidar_obs = obs.copy()
        self._lidar_vel = lidar_data_vel.copy()
        self._text_obs = text_embed.copy()

        lidar_obs_env_torch = torch.as_tensor(self._lidar_obs).to(self.model_device).type(torch.float).to(
            self.model_device)
        text_obs_env_torch = torch.as_tensor(self._text_obs).to(self.model_device).type(torch.float)

        self._latent_state_env = self.robot_model.get_state_representation(lidar_obs_env_torch, text_obs_env_torch)

        lidar_obs_agent_torch = torch.as_tensor(self._lidar_obs).to(self.model_device).type(torch.float)
        text_obs_agent_torch = torch.as_tensor(self._text_obs).to(self.model_device).type(torch.float)

        human_model = self.mirror_model

        self._latent_state_agent = human_model.get_state_representation(lidar_obs_agent_torch, text_obs_agent_torch)

        self._action, action_agent_dist = human_model.policy(self._latent_state_agent.unsqueeze(dim=0))
        self._action = torch.tanh(self._action[0])
        self._latent_state_agent = self._latent_state_agent.detach().cpu().numpy()
        self._latent_state_env = self._latent_state_env.detach().cpu().numpy()

        self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        return self._state.copy(), collided, done

    def reset(self, ego_car=None, other_car=None, speed_up_speed=None, rear_spawn_pos=None, front_spawn_pos=None, slow_down_speed=None):
        self.resetPlayer(ego_car, other_car)
        self.world.tick()
        self.resetOtherVehicles(speed_up_speed, rear_spawn_pos, front_spawn_pos, slow_down_speed)
        self.world.tick()
        self.collision_intensity = []
        self.init_accel = False
        self.prev_text = ""
        obs, lidar_data_id, lidar_data_vel, reward, collided, done = self.simulator_step(action=None, frame_count=0)
        text_embed = generate_text_embed(self.text_model, obs, lidar_data_vel, self.model_device)

        self._lidar_obs = obs.copy()
        self._lidar_vel = lidar_data_vel.copy()
        self._text_obs = text_embed.copy()

        lidar_obs_env_torch = torch.as_tensor(self._lidar_obs).to(self.model_device).type(torch.float).to(
            self.model_device)
        text_obs_env_torch = torch.as_tensor(self._text_obs).to(self.model_device).type(torch.float)

        self._latent_state_env = self.robot_model.get_state_representation(lidar_obs_env_torch, text_obs_env_torch)

        lidar_obs_agent_torch = torch.as_tensor(self._lidar_obs).to(self.model_device).type(torch.float)
        text_obs_agent_torch = torch.as_tensor(self._text_obs).to(self.model_device).type(torch.float)

        human_model = self.mirror_model
        self._latent_state_agent = human_model.get_state_representation(lidar_obs_agent_torch, text_obs_agent_torch)

        self._action, action_agent_dist = human_model.policy(self._latent_state_agent.unsqueeze(dim=0))
        self._action = torch.tanh(action_agent_dist.mean[0])
        self._latent_state_agent = self._latent_state_agent.detach().cpu().numpy()
        self._latent_state_env = self._latent_state_env.detach().cpu().numpy()

        self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

        self.text_p = None
        self.lidar_p = None

        if self.forward_record_flag and self.demonstration_count > 0:
            print(f'save human demonstrations with size {self.demonstration_count}')
            np.save(f'{self.demonstration_save_path}/lidar_obs', self.lidar_obs_buffer[:self.demonstration_count])
            np.save(f'{self.demonstration_save_path}/text_embed', self.text_embed_buffer[:self.demonstration_count])
            np.save(f'{self.demonstration_save_path}/human_actions',
                    self.human_actions_buffer[:self.demonstration_count])
            np.save(f'{self.demonstration_save_path}/latent_state', self.latent_state_buffer[:self.demonstration_count])

        # Store global positions and speed for all cars
        self.car_poses = np.zeros((5, 6))
        self.car_speeds = np.zeros((5, 3))
        for i in range(self.car_poses.shape[0]):
            if i < 4:
                vehicle = self.world.get_actor(self.vehicles_list[i])
                vehicle_tf = vehicle.get_transform()
                vehicle_vel = vehicle.get_velocity()
            else:
                vehicle_tf = self.player.get_transform()
                vehicle_vel = self.player.get_velocity()
            self.car_poses[i, 0] = vehicle_tf.location.x
            self.car_poses[i, 1] = vehicle_tf.location.y
            self.car_poses[i, 2] = vehicle_tf.location.z
            self.car_poses[i, 3] = vehicle_tf.rotation.roll
            self.car_poses[i, 4] = vehicle_tf.rotation.pitch
            self.car_poses[i, 5] = vehicle_tf.rotation.yaw
            self.car_speeds[i, 0] = vehicle_vel.x
            self.car_speeds[i, 1] = vehicle_vel.y
            self.car_speeds[i, 2] = vehicle_vel.z

        return self._state.copy(), collided, done, self._lidar_obs


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
        curr_yaw = vehicle_waypoint_closest_to_road.transform.rotation.yaw
        if curr_yaw < 0:
            curr_yaw += 360.0
        if next_goal_waypoint1[0].transform.rotation.yaw < 0:
            next_goal_waypoint1[0].transform.rotation.yaw += 360.0
        forward_locations = np.array([next_goal_waypoint1[0].transform.rotation.yaw - curr_yaw])
        for i in range(forward_locations.shape[0]):
            if forward_locations[i] <= -180.0:
                forward_locations[i] += 360.0
            elif forward_locations[i] > 180.0:
                forward_locations[i] -= 360.0
        forward_locations /= 180.0
        return forward_locations

    def step_communication(self, action, condition_count, episode):
        rewards = []
        for _ in range(self.frame_skip):
            self._state, reward, collided, done, human_action, lidar_obs = self.forward(self._state, action)
            rewards.append(reward)
            if done or collided:
                break

        return self._state.copy(), np.mean(rewards), collided, done, human_action, lidar_obs

    def forward(self, current_state: np.array, action_mask: np.array):
        self.lidars_cars_id = [[0, 0, 0, 0]] * 3
        self.lidars_i = 0
        # Update the latent state first, then sample action to forward environment
        with torch.no_grad():
            self._action_mask = action_mask.copy()
            action_mask_torch = torch.tensor(self._action_mask).to(self.model_device).type(torch.float)  # .to('cpu')

            print(self._lidar_obs[1])  # , self._lidar_obs[1+3], self._lidar_obs[36*3-1-1])
            lidar_obs_env_torch = torch.as_tensor(self._lidar_obs).to(self.model_device).type(torch.float)
            text_obs_env_torch = torch.as_tensor(self._text_obs).to(self.model_device).type(torch.float)

            lidar_action_mask_torch, text_action_mask_torch = seperate_lidar_text_mask(action_mask_torch)
            limited_view_mask = torch.ones_like(lidar_action_mask_torch).to(self.model_device)
            limited_view_mask[..., 0:8 * 3] = 0.0
            limited_view_mask[..., 28 * 3:36 * 3] = 0.0

            self._latent_state_env = self.robot_model.get_state_representation(lidar_obs_env_torch,
                                                                               text_obs_env_torch,
                                                                               action=torch.as_tensor(self._action).to(
                                                                                   self.model_device).type(torch.float),
                                                                               pre_state=torch.as_tensor(
                                                                                   self._latent_state_env).to(
                                                                                   self.model_device).type(torch.float),
                                                                               lidar_mask=lidar_action_mask_torch * 0.0 + 1.0,
                                                                               text_mask=text_action_mask_torch * 0.0)

            # print(self.get_q_value(self.robot_model, self._latent_state_env).item())
            self.pre_q_value = self.get_q_value(self.robot_model, self._latent_state_env).item()
            text_embed_torch = self.robot_model.autoencoder.decoder_text(self._latent_state_env).mean

            lidar_obs_2d = lidar_obs_env_torch[..., :-5][..., 1::3]
            lidar_obs_mask = self.perceptual_mask.get_obs_mask(lidar_obs_2d)
            # print(lidar_obs_mask.detach().cpu().numpy())
            lidar_obs_mask_rsample_list = [lidar_obs_mask[k:k + 1].repeat(3) for k in range(36)]

            lidar_obs_mask = torch.cat(lidar_obs_mask_rsample_list +
                                       [torch.ones_like(lidar_obs_mask[0:0 + 1]).repeat(5).to(
                                           self.model_device)], dim=-1) # * 0.0

            lidar_mask = (lidar_obs_mask + lidar_action_mask_torch * limited_view_mask > 0.5).type(torch.float)

            human_model = self.mirror_model


            self._latent_state_agent = human_model.get_state_representation(lidar_obs_env_torch,
                                                                            text_embed_torch,  # text_obs_env_torch,
                                                                            action=torch.as_tensor(self._action).to(
                                                                                self.model_device).type(torch.float),
                                                                            pre_state=torch.as_tensor(
                                                                                self._latent_state_agent).to(
                                                                                self.model_device).type(torch.float),
                                                                            lidar_mask=lidar_mask,
                                                                            text_mask=text_action_mask_torch)

            text_embed = torch.stack([text_embed_torch[..., 0 * 20:1 * 20 - 1],
                                      text_embed_torch[..., 1 * 20:2 * 20 - 1],
                                      text_embed_torch[..., 2 * 20:3 * 20 - 1],
                                      text_embed_torch[..., 3 * 20:4 * 20 - 1],
                                      text_embed_torch[..., 4 * 20:5 * 20 - 1],
                                      text_embed_torch[..., 5 * 20:6 * 20 - 1]], dim=0) * 15.0
            text_distance = torch.stack([text_embed_torch[..., 1 * 20 - 1],
                                         text_embed_torch[..., 2 * 20 - 1],
                                         text_embed_torch[..., 3 * 20 - 1],
                                         text_embed_torch[..., 4 * 20 - 1],
                                         text_embed_torch[..., 5 * 20 - 1],
                                         text_embed_torch[..., 6 * 20 - 1]], dim=0)

            text_list = []
            text_distance_list = []
            for i in range(6):
                if text_action_mask_torch[i * 20].item() == 1:
                    gpt_embed = self.text_model.decoder_text(text_embed[i]).mean
                    prediction = self.gpt_lm_model[i](gpt_embed.reshape(-1, 768))
                    predicted_text = ""
                    for k in range(13):
                        predicted_index = torch.argmax(prediction.reshape(13, -1)[k, :]).item()
                        word = self.tokenizer.decode(predicted_index)
                        if word != '!':
                            predicted_text += word
                    print(predicted_text)
                    text_list += [predicted_text]
                    text_distance_list += [text_distance[i]]

            if len(text_list) > 0:
                text_id = np.random.randint(low=0, high=len(text_list), size=(1,)).item()
                text = text_list[text_id]
                text_d = text_distance_list[text_id]
                dir_path = os.path.dirname(os.path.realpath(__file__))
                # print(dir_path)
                if text != "":
                    if time.time() - self.speech_time > 3.5:
                        try:
                            # print(time.time() - self.speech_time)
                            # print(f'{text_d}, {text[1:]}')
                            play(file=dir_path[:-9] + 'utils/robot_comm_audio/' + text[1:] + '.wav', volume=0.0)
                            if 0.25 <= text_d < 0.75:
                                # print('nearby')
                                play(file=dir_path[:-9] + 'utils/robot_comm_audio/' + 'beep.wav', volume=0.5)
                            elif 0.0 <= text_d < 0.25:
                                # print('close')
                                play(file=dir_path[:-9] + 'utils/robot_comm_audio/' + 'beep.wav', volume=1.0)
                            else:
                                # print('no car')
                                play(file=dir_path[:-9] + 'utils/robot_comm_audio/' + 'beep.wav', volume=0.1)
                            time.sleep(0.2)
                            self.speech_time = time.time()
                            play(file=dir_path[:-9] + 'utils/robot_comm_audio/' + text[1:] + '.wav')
                            self.speech = text
                            self.prev_text = text
                            # self.text_record =
                        except:
                            pass
            else:
                self.speech = ''

            if self.steering_agent is not None:
                if self.steering_agent.is_wheel:
                    # Physical action using the steering wheel
                    self._action = self.steering_agent.parseVehicleWheel()
                    self._action = torch.tensor([self._action.throttle, self._action.steer])
                else:
                    # Physical action using the keyboard
                    self._action = self.steering_agent.parseVehicleKey(self.clock.get_time())
                    self._action = torch.tensor([self._action.throttle, self._action.steer])
            # Check if other vehicles can start accelerating (based on throttle)
            if not self.init_accel:
                if self._action[0].item() > 0.0:
                    self.accelOtherVehicles()
                    self.init_accel = True

            self._latent_state_agent = self._latent_state_agent.detach().cpu().numpy()
            self._latent_state_env = self._latent_state_env.detach().cpu().numpy()
            self._state = np.concatenate([self._latent_state_agent, self._latent_state_env], axis=-1)

            for i in range(3):
                self._lidar_obs, _, self._lidar_vel, reward, collided, done = self.simulator_step(
                    self._action.detach().cpu().numpy(), i)
                if done or collided:
                    break
            text_embed = generate_text_embed(self.text_model, self._lidar_obs, self._lidar_vel, self.model_device)
            self._text_obs = text_embed.copy()

            if self.forward_record_flag and self.record_demonstration_flag:  # and (self._lidar_obs[1] < 0.95 or self._lidar_obs[3 * 18 + 1] < 0.95):
                self.lidar_obs_buffer[self.demonstration_count] = self._lidar_obs.copy()
                self.text_embed_buffer[self.demonstration_count] = self._text_obs.copy()
                # print(self._text_obs)
                self.latent_state_buffer[self.demonstration_count] = self._latent_state_agent
                self.human_actions_buffer[self.demonstration_count] = self._action.detach().cpu().numpy().copy()
                self.demonstration_count += 1
            return self._state, reward, collided, done, self._action.detach().cpu().numpy(), self._lidar_obs.copy()

    def step(self, action):
        rewards = []
        for _ in range(self.frame_skip):
            # Check if other vehicles can start accelerating (based on throttle)
            if self.expt_stage == 'warmup-fog' and not self.init_accel:
                if action[0].item() > 0.0:
                    self.accelOtherVehicles()
                    self.init_accel = True
            next_obs, lidar_data_id, lidar_data_vel, reward, collided, done = self.simulator_step(action, _)
            rewards.append(reward)
            if done or collided:
                break
        return next_obs, lidar_data_id, lidar_data_vel, np.mean(rewards), collided, done

    def simulator_step(self, action, frame_count):
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
        # if frame_count == 0 or frame_count == 1
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
                self.traffic_manager.vehicle_percentage_speed_difference(front_car, self.slow_down_speed)
                self.traffic_manager.vehicle_percentage_speed_difference(back_car, self.slow_down_speed)
        elif self.speed_up_down == 4:  # down-right
            checkpt = self.map.get_waypoint_xodr(road_id=22, lane_id=-3, s=self.slow_down_pos).transform.location
            front_car = self.world.get_actor(self.vehicles_list[3])
            if np.linalg.norm(np.array([checkpt.x - front_car.get_location().x,
                                        checkpt.y - front_car.get_location().y])) < 5.0:
                back_car = self.world.get_actor(self.vehicles_list[1])
                vehicle_control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.1, hand_brake=False,
                    reverse=False, manual_gear_shift=False)
                front_car.apply_control(vehicle_control)
                back_car.apply_control(vehicle_control)
                self.traffic_manager.vehicle_percentage_speed_difference(front_car, self.slow_down_speed)
                self.traffic_manager.vehicle_percentage_speed_difference(back_car, self.slow_down_speed)

        lidar_data_3d, lidar_data_2d, lidar_data_id, lidar_data_vel, lidar_type = process_lidar(self.lidar_scan, self)

        lidar_data_3d = np.reshape(lidar_data_3d, (lidar_data_3d.shape[0] * lidar_data_3d.shape[1],))
        # Render display
        if self.render_display:
            # draw_combined_image(self.display, rgb_image_front, rgb_image_back)
            # draw_front_image(self.display, rgb_image_front)
            self.draw_fpv_image(self, self.rgb_image_front, self.rgb_image_back, self.render_fog, self.ready_mode,
                                self.survey_mode)

            if self._action_mask is not None:
                lidar_action_mask, _ = seperate_lidar_text_mask(self._action_mask)
                lidar_action_mask = lidar_action_mask[..., :36 * 3]
                lidar_action_mask = lidar_action_mask[..., 1:][..., ::3]

                self.visualizeCars(lidar_data_2d, lidar_type, lidar_action_mask)
            pygame.display.flip()

        next_obs = self.get_obs()
        next_obs = np.concatenate((lidar_data_3d, next_obs), axis=0)
        reward, done, collided = self.getReward(steer, brake, lidar_data_2d, lidar_type, next_obs[-2])

        # Store global positions and speed for all cars
        self.car_poses = np.zeros((5, 6))
        self.car_speeds = np.zeros((5, 3))
        for i in range(self.car_poses.shape[0]):
            if i < 4:
                vehicle = self.world.get_actor(self.vehicles_list[i])
                vehicle_tf = vehicle.get_transform()
                vehicle_vel = vehicle.get_velocity()
            else:
                vehicle_tf = self.player.get_transform()
                vehicle_vel = self.player.get_velocity()
            self.car_poses[i, 0] = vehicle_tf.location.x
            self.car_poses[i, 1] = vehicle_tf.location.y
            self.car_poses[i, 2] = vehicle_tf.location.z
            self.car_poses[i, 3] = vehicle_tf.rotation.roll
            self.car_poses[i, 4] = vehicle_tf.rotation.pitch
            self.car_poses[i, 5] = vehicle_tf.rotation.yaw
            self.car_speeds[i, 0] = vehicle_vel.x
            self.car_speeds[i, 1] = vehicle_vel.y
            self.car_speeds[i, 2] = vehicle_vel.z

        # Check if ego car has reached goal line
        if self.expt_stage != 'warmup-clear':
            player_loc = self.player.get_location()
            for goal_loc in self.all_goal_lanes_loc:
                if np.linalg.norm(np.array([(goal_loc.x - player_loc.x, goal_loc.y - player_loc.y)])) < 2.0:
                    done = True

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

    def visualizeCars(self, lidar_data, lidar_type, lidar_action_mask):
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
                vehicle_angle_wrt_world = np.pi - vehicle_angle_wrt_world
            elif vehicle_loc_wrt_player.x < 0. and vehicle_loc_wrt_player.y < 0.:
                vehicle_angle_wrt_world = -np.pi + vehicle_angle_wrt_world
            vehicle_angle_wrt_player = vehicle_angle_wrt_world - player_angle

            nearest_beam_index = int(np.round((vehicle_angle_wrt_player + np.pi) / (2. * np.pi) * 36.))
            if 0 <= nearest_beam_index < 36:
                # print("test!!!!", lidar_type[nearest_beam_index], nearest_beam_index, vehicle_angle_wrt_player)
                if lidar_action_mask[nearest_beam_index] == 1 and \
                        lidar_type[nearest_beam_index] == 0:
                    cars.append(vehicle)
                    for i, id in enumerate(self.vehicles_list):
                        if vehicle.id == id:
                            self.lidars_cars_id[self.lidars_i][i] = 1
                    self.lidars_i += 1

        b_boxes = get_bounding_boxes(cars, self.camera_rgb_front, 'car')
        draw_bounding_boxes(self.display, b_boxes, 'car')

    def draw_fpv_image(self, env, image_front, image_back, render_fog,
                       is_ready, is_survey):
        front_array = np.frombuffer(image_front.raw_data, dtype=np.dtype("uint8"))
        back_array = np.frombuffer(image_back.raw_data, dtype=np.dtype("uint8"))
        front_array = np.reshape(front_array, (image_front.height, image_front.width, 4))
        back_array = np.reshape(back_array, (image_back.height, image_back.width, 4))
        front_array = front_array.copy()
        back_array = back_array.copy()
        # Ready mode
        if is_ready or is_survey:
            front_array[:, :, :-1] = (front_array[:, :, :-1] / 4).astype(np.int)
            back_array[:, :, :-1] = (back_array[:, :, :-1] / 4).astype(np.int)
        if render_fog and not is_ready:
            front_array[:, :, :-1] = (front_array[:, :, :-1] / 1.2).astype(np.int)
        front_array = front_array[:, :, :3]
        back_array = back_array[:, :, :3]
        front_array = front_array[:, :, ::-1]
        back_array = back_array[:, :, ::-1]

        if render_fog:
            front_blur_array = self.front_blur_mask * self.blank_front + (1 - self.front_blur_mask) * front_array.copy()
            front_array = self.front_mask * front_blur_array + (1 - self.front_mask) * front_array
            front_array = front_array.astype(np.uint8)

            back_blur_array = self.back_blur_mask * self.blank_back + (1 - self.back_blur_mask) * back_array.copy()
            back_array = self.back_mask * back_blur_array + (1 - self.back_mask) * back_array
            back_array = back_array.astype(np.uint8)
            back_array = cv2.blur(back_array, (7, 7), 0)
        v_back = back_array.copy()
        cv2.flip(back_array, 1, v_back)

        image_surface = pygame.surfarray.make_surface(front_array.swapaxes(0, 1))
        env.display.blit(image_surface, (0, 0))
        image_surface = pygame.surfarray.make_surface(v_back.swapaxes(0, 1))
        env.display.blit(image_surface, (220, 30))  # (370, 30))#(610, 60))
        # Display ready mode
        if is_ready:
            myfont = pygame.font.SysFont('Comic Sans MS', 24)
            if env.expt_stage != 'expt':
                text = myfont.render('Warm Up Sessions', False, (255, 255, 255))
            else:
                text = myfont.render('Actual Sessions', False, (255, 255, 255))
            env.display.blit(text, (int(image_front.width / 3) + 50, int(image_front.height / 2) - 200))
            if env.expt_stage == 'warmup-clear':
                text = myfont.render('Clear Weather', False, (255, 255, 255))
                env.display.blit(text, (int(image_front.width / 3) + 70, int(image_front.height / 2) - 150))
            else:
                text = myfont.render('Round ' + str(env.episode + 1) + ' / ' + str(env.total_episodes), False,
                                     (255, 255, 255))
                env.display.blit(text, (int(image_front.width / 3) + 80, int(image_front.height / 2) - 150))
            text = myfont.render('Press the red button on steering wheel to begin', False, (255, 255, 255))
            env.display.blit(text, (int(image_front.width / 3) - 140, int(image_front.height / 2)))
        elif is_survey:
            myfont = pygame.font.SysFont('Comic Sans MS', 24)
            text = myfont.render('Please complete the short survey before continuing', False, (255, 255, 255))
            env.display.blit(text, (int(image_front.width / 3) - 150, int(image_front.height / 2) - 50))
            text = myfont.render('Press the red button on steering wheel to continue', False, (255, 255, 255))
            env.display.blit(text, (int(image_front.width / 3) - 150, int(image_front.height / 2) + 50))
        # Display wall during normal mode
        if not is_ready:
            player_location = env.player.get_location()
            walls = [wall for wall in env.all_walls if
                     (np.linalg.norm(np.array([wall.bounding_box.location.x, wall.bounding_box.location.y]) - np.array(
                         [player_location.x, player_location.y])) < 40.)]
            b_boxes = get_bounding_boxes(walls, env.camera_rgb_front, 'wall')
            draw_bounding_boxes(env.display, b_boxes, 'wall')
        # Display goal line
        if env.expt_stage != 'warmup-clear':
            debug = env.world.debug
            debug.draw_line(begin=carla.Location(x=env.goal_left_loc.x, y=env.goal_left_loc.y, z=1.0),
                            end=carla.Location(x=env.goal_right_loc.x, y=env.goal_right_loc.y, z=1.0),
                            thickness=2.0,
                            color=carla.Color(0, 255, 0))
