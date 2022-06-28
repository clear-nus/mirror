    #!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

# Import CARLA modules
import glob
import os
import sys

import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# Import miscellaneous
import pygame
import argparse
import logging
import torch
import os

from utils.audio_player import play, stop

from envs.expt.carla_env import CarlaEnv
from envs.steering_wheel_control import SteeringWheelControl
from models.comm_cat_model import BisimRLModel
from models.cat_model import BisimModel, PerceptualMaskDis, PolicyFilter

from algos.implants_algo import train_implants

from utils.exp_data_recoder import DataRecorder

torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

def game_loop(args):
    """ Main loop for agent"""
    # Initializations
    env = None
    try:
        steering_control = SteeringWheelControl(is_wheel=True)

        robot_model = BisimModel(action_shape=(2,)).to(args.device)
        robot_model.load_state_dict(torch.load('./current_model/model_latest.pt'))
        robot_model.eval()
        robot_model.to(args.device)
        mirror_model = robot_model

        rl_model = BisimRLModel(action_shape=(2 * 6,)).to(args.device)
        rl_model.eval()
        env = CarlaEnv(render_display=args.render, render_fog=True, expt_stage='expt',
                       host=args.host, port=args.port,
                       tm_port=args.tm_port, frame_skip=1,
                       image_resolution=(args.width, args.height),
                       robot_model=robot_model,
                       mirror_model=mirror_model,
                       steering_agent=steering_control,
                       forward_record_flag=False)

        # Main Expt
        exp_cases_template = [('l', 'lf'),
                              ('l', 'lr'),
                              ('l', 'rr'),
                              ('r', 'lf'),
                              ('r', 'rf'),
                              ('r', 'rr')]

        # generate save path for each participant, given their id
        save_path = f'./exp_data/participant_{args.id}'
        if os.path.isdir(save_path) is not True:
            os.mkdir(save_path)
        env.demonstration_save_path = save_path

        exp_data_recoder = DataRecorder(save_path)

        condition_order = ['noop', 'mirror']
        condition_length = len(exp_cases_template)

        env.expt_stage = 'expt'
        env.episode = 0

        env.total_episodes = condition_length

        itr = 0
        episode_count = 0.0

        exp_cases = exp_cases_template.copy()
        # For each condition
        for condition_count, condition in enumerate(condition_order):
            print('Phase', condition_count, '(' + condition + ')')
            env.condition = condition
            perceptual_mask = PerceptualMaskDis()
            policy_filter = PolicyFilter(robot_model.latent_size)
            perceptual_mask.load_state_dict(torch.load(f'{save_path}/perceptual_mask.pt'))
            policy_filter.load_state_dict(torch.load(f'{save_path}/policy_filter.pt'))
            env.set_perceptual_mask(perceptual_mask)
            env.set_policy_filter(policy_filter)

            # For each episode
            while True:
                if env.ready_mode:

                    exp_case = exp_cases.pop()
                    print(exp_case, exp_cases, env.episode)
                    while True:
                        obs, _, done, lidar_obs = env.reset(ego_car=exp_case[0], other_car=exp_case[1])
                        if not done:
                            break
                    exp_data_recoder.add_case_setup(front_spawn_pos=env.front_spawn_pos,
                                                    rear_spawn_pos=env.rear_spawn_pos,
                                                    episode_num=env.episode,
                                                    rear_speed=env.speed_up_speed,
                                                    front_speed=env.slow_down_speed)
                    ready_flag = False
                # Wait for red button to be pressed at the start
                while not ready_flag:
                    if steering_control.parseKey() or steering_control.parseButton():
                        ready_flag = True
                        env.ready_mode = False
                # Start the episode
                # Generate communication action according to the condition
                if (condition == 'warmup' or condition == 'noop') and episode_count > 20:
                    env.forward_record_flag = True
                else:
                    env.forward_record_flag = False

                if condition == 'warmup' or condition == 'noop':
                    if episode_count == 0 or episode_count % 1 == 0:
                        opti_action_mask_torch, opti_lidar_action_mask_torch, opti_text_action_mask_torch, \
                        lidar_action_mask_np, text_action_mask_np = env.adaptive_random_shooting()
                    obs = torch.tensor(obs).float().to(args.device)
                    control, control_dist = rl_model.policy(obs)
                    control = (control > 0.5).type(torch.float)
                    lidar_action_mask = control[..., :6] * 0.0
                    text_action_mask = control[..., 6:] * 0.0
                    lidar_action_mask_np = np.zeros(6)
                    text_action_mask_np = np.zeros(6)

                    lidar_action_mask = [lidar_action_mask[..., 0:1].repeat(3 * 1) * 1.0,
                                         lidar_action_mask[..., 1:2].repeat(3 * 8) * 1.0,
                                         lidar_action_mask[..., 2:3].repeat(3 * 9),
                                         lidar_action_mask[..., 3:4].repeat(3 * 1),
                                         lidar_action_mask[..., 4:5].repeat(3 * 9),
                                         lidar_action_mask[..., 5:6].repeat(3 * 8) * 1.0,
                                         torch.ones_like(lidar_action_mask[..., 0:1]).repeat(5).to(control.device).type(
                                             torch.float)]
                    lidar_action_mask = torch.cat(lidar_action_mask, dim=-1).detach().cpu().numpy()

                    text_action_mask = [text_action_mask[..., 0:1].repeat(20),
                                        text_action_mask[..., 1:2].repeat(20),
                                        text_action_mask[..., 2:3].repeat(20),
                                        text_action_mask[..., 3:4].repeat(20),
                                        text_action_mask[..., 4:5].repeat(20),
                                        text_action_mask[..., 5:6].repeat(20)]
                    text_action_mask = torch.cat(text_action_mask, dim=-1).detach().cpu().numpy()

                    control_expand = np.concatenate([lidar_action_mask, text_action_mask], axis=-1)
                else:
                    if episode_count == 0 or episode_count % 1 == 0:
                        opti_action_mask_torch, opti_lidar_action_mask_torch, opti_text_action_mask_torch, \
                        lidar_action_mask_np, text_action_mask_np = env.adaptive_random_shooting()
                    control_expand = opti_action_mask_torch.detach().cpu().numpy()

                if episode_count < 20:
                    control_expand[:] = 0.0
                    control_expand[3 * 36:3 * 36 + 5] = 1.0

                next_obs, reward, collided, done, human_action, lidar_obs = env.step_communication(control_expand,
                                                                                                   condition_count,
                                                                                                   env.episode)

                if collided:
                    play(file='./utils/robot_comm_audio/collided.wav')
                exp_data_recoder.add_data(task_reward=reward, speed=next_obs[..., -5:], collision=collided,
                                          comm_lidar_action=lidar_action_mask_np,
                                          comm_text_action=text_action_mask_np,
                                          human_action=human_action,
                                          global_speed=env.car_speeds,
                                          car_pos=env.car_poses,
                                          lidar_car_id=env.lidars_cars_id,
                                          speech=env.speech)
                episode_count += 1

                obs = next_obs

                if done or collided:  # If episode ended or collided
                    itr += 1
                    env.episode += 1

                    # save exp data
                    exp_data_recoder.clear_buffer()

                    if env.episode == condition_length:
                        env.survey_mode = True
                        exp_cases = exp_cases_template.copy()
                        while True:
                            obs, _, done, lidar_obs = env.reset()
                            if not done:
                                break
                        survey_flag = False

                        # if current condition is noop, train models on human data
                        # train perceptual_mask
                        if condition == 'warmup' or condition == 'noop':
                            perceptual_mask, policy_filter = train_implants(save_path)
                            env.set_perceptual_mask(perceptual_mask)
                            env.set_policy_filter(policy_filter)

                        while not survey_flag:
                            if steering_control.parseKey() or steering_control.parseButton():
                                survey_flag = True
                                env.survey_mode = False
                        env.episode = 0
                        env.ready_mode = True
                        env.init_accel = False
                        print('Steps:', episode_count)
                        episode_count = 0.0
                        break
                    env.ready_mode = True
                    env.init_accel = False
                    print('Steps:', episode_count)
                    episode_count = 0.0

                    if env.episode == condition_length:
                        env.episode = 0
                        break

    finally:
        if env is not None:
            env.destroy()

        pygame.quit()
        print("All of all rounds. Stopping program.")


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--render',
        default=True,
        type=bool,
        help='Render display of 3rd person view (default: True)'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--device',
        help='GPU device',
        default=0,
        type=int)
    argparser.add_argument(
        '--condition',
        help='om/ase',
        default='om',
        type=str
    )
    argparser.add_argument('--id',
                           help='participant id',
                           default=0,
                           type=int)
    argparser.add_argument('--condition_id',
                           help='Condition to start from in case of program crash (0 or 1 or 2)',
                           default=0,
                           type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.device = torch.device('cuda', args.device)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
