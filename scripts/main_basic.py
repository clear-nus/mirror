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
import random
import sys

import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Set to correct path to import agents
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
from carla.agents.navigation.multimodal_behavior_agent import BehaviorAgentBackCar

# Import miscellaneous
import pygame
import argparse
import logging
import torch

from algos.cat_basic_algo import Bisimulation
from envs.carla_env_basic import CarlaEnvBackCar
from models.cat_model import BisimModel
from models.text_model import AutoEncodersUni
from utils.replay_buffer import ReplayBufferText


def generate_text_embed(text_model, obs: torch.Tensor, speed, device="cuda:1"):
    spped_threshold = 32
    detect_threshold = 0.8
    front_lidar = torch.tensor([obs[3 * 18 + 1]]).to(torch.float)
    front_speed = torch.tensor([0]).to(torch.float)
    if front_lidar < detect_threshold:
        front_speed = torch.tensor([speed[18]])
    h_dir = 'front'
    v_dir = 'front'
    if front_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)
    front_text_embed = text_model.get_latent_dist(front_lidar, speed_value, h_dir, v_dir).mean.detach().cpu().numpy() / 15.0
    front_text_embed = np.concatenate([front_text_embed, front_lidar], axis=-1)

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
    if left_front_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)
    left_front_text_embed = text_model.get_latent_dist(left_front_lidar, speed_value, h_dir, v_dir).mean.detach().cpu().numpy() / 15.0
    left_front_text_embed = np.concatenate([left_front_text_embed, left_front_lidar], axis=-1)

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
    if right_front_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)
    right_front_text_embed = text_model.get_latent_dist(right_front_lidar, speed_value,
                                                        h_dir, v_dir).mean.detach().cpu().numpy() / 15.0
    right_front_text_embed = np.concatenate([right_front_text_embed, right_front_lidar], axis=-1)

    # generate h_dir == 'rear'
    rear_lidar = torch.tensor([obs[3 * 0 + 1]]).to(torch.float)
    rear_speed = torch.tensor([0]).to(torch.float)
    if rear_lidar < detect_threshold:
        rear_speed = torch.tensor([speed[0]]).to(torch.float)
    h_dir = 'rear'
    v_dir = 'rear'
    if rear_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)
    rear_text_embed = text_model.get_latent_dist(rear_lidar, speed_value, h_dir, v_dir).mean.detach().cpu().numpy() / 15.0
    rear_text_embed = np.concatenate([rear_text_embed, rear_lidar], axis=-1)

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
    if left_rear_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)
    left_rear_text_embed = text_model.get_latent_dist(left_rear_lidar, speed_value, h_dir, v_dir).mean.detach().cpu().numpy() / 15.0
    left_rear_text_embed = np.concatenate([left_rear_text_embed, left_rear_lidar], axis=-1)

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
    if right_rear_speed > spped_threshold:
        speed_value = torch.tensor([1]).to(torch.float)
    else:
        speed_value = torch.tensor([0]).to(torch.float)
    right_rear_text_embed = text_model.get_latent_dist(right_rear_lidar, speed_value, h_dir, v_dir).mean.detach().cpu().numpy() / 15.0
    right_rear_text_embed = np.concatenate([right_rear_text_embed, right_rear_lidar], axis=-1)

    text_embed = np.concatenate([front_text_embed, left_front_text_embed, right_front_text_embed,
                                 rear_text_embed, left_rear_text_embed, right_rear_text_embed], axis=-1)
    # return np.zeros(20 * 6)
    return text_embed


def game_loop(args):
    """ Main loop for agent"""
    tot_target_reached = 0
    num_min_waypoints = 5

    # Initializations
    env = None
    try:

        is_wheel_control = False
        steering_control = None

        model = BisimModel(action_shape=(2,)).to(args.device)
        text_model = AutoEncodersUni(latent_size=19, hidden_size=64, layers=3)
        text_model = torch.load('./current_model/autoencoder.pt')
        text_model.set_device('cpu')
        model.load_state_dict(torch.load('./current_model/model_latest.pt'))
        algo = Bisimulation(model=model, device=args.device)

        obs_shape = (36 * 3 + 5,)
        env = CarlaEnvBackCar(render_display=args.render, render_fog=args.render_fog,
                              host=args.host, port=args.port,
                              tm_port=args.tm_port, frame_skip=3,
                              image_resolution=(args.width, args.height))
        agent = BehaviorAgentBackCar(env.player, env.vehicles_list, behavior='normal')

        obs, lidar_data_id, lidar_data_vel, _, _, _ = env.simulator_step(action=None)

        text_embed = generate_text_embed(text_model, obs, lidar_data_vel)
        replay_buffer = ReplayBufferText(obs_shape=obs_shape,
                                         text_shape=(20 * 6,),
                                         action_shape=(2,),
                                         reward_shape=(1,),
                                         capacity=100000,
                                         batch_size=500,
                                         device=torch.device('cuda', 1))

        algo.optim_initialize(model)

        collect_count = 0
        itr = 0
        task_reward = 0.0
        episode_count = 0.0
        beta, beta_decay = 0.0, 1.0
        expert_flag = False
        explore_flag = True

        curr_waypoint = env.map.get_waypoint(agent.vehicle.get_location())
        destination = curr_waypoint.next(100.)[0].transform.location
        agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        state = None
        control = None
        while True:
            # Set new destination when target has been reached
            if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints:
                agent.reroute(env)
                tot_target_reached += 1
            elif len(agent.get_local_planner().waypoints_queue) == 0:
                print("Target reached, mission accomplished...")
                break
            agent.update_information()
            expert_control = agent.run_step()
            # Send model command
            # Run neural network with observation input
            obs = torch.tensor(obs).float().to(args.device)
            text_embed = torch.tensor(text_embed).float().to(args.device)

            if expert_flag:
                control = np.array([expert_control.throttle, expert_control.steer])
            else:
                if control is not None:
                    control_torch = torch.tensor(control).float().to(args.device)
                    if explore_flag:
                        temp_lidar_mask = torch.zeros_like(obs)
                        temp_lidar_mask[...,-5:] = 1.0
                        state = model.get_state_representation(obs,
                                                               text_embed,
                                                               control_torch,
                                                               state,
                                                               lidar_mask=temp_lidar_mask,
                                                               text_mask=torch.zeros_like(text_embed))
                    else:
                        state = model.get_state_representation(obs,
                                                               text_embed,
                                                               control_torch,
                                                               state)
                else:
                    state = model.get_state_representation(obs, text_embed)
                _, control_dist = model.policy(state)
                control = control_dist.rsample()
                # Tanh post-processing to generate control command
                if np.random.rand(1) < 0.1:
                    noise = torch.rand(size=control.size()).to(control.device) * 0.05
                    control = torch.tanh(control + noise).detach().cpu().numpy()
                else:
                    control = torch.tanh(control).detach().cpu().numpy()
            if steering_control != None:
                if is_wheel_control:
                    # Physical action using the steering wheel
                    control = steering_control.parseVehicleWheel()
                    control = torch.tensor([control.throttle, control.steer])
                else:
                    # Physical action using the keyboard
                    control = steering_control.parseVehicleKey(env.clock.get_time())
                    control = torch.tensor([control.throttle, control.steer])
            # Execute model command in env
            next_obs, next_lidar_data_id, next_lidar_data_vel, reward, collided, done = env.step(control)
            next_text_embed = generate_text_embed(text_model, next_obs, next_lidar_data_vel)

            task_reward += reward
            episode_count += 1

            expert_control = np.array([expert_control.throttle, expert_control.steer])
            reward, done = np.array([reward]), np.array([done])
            if not env.player.get_location().z > 0.5:
                replay_buffer.add(next_obs, next_text_embed, control, expert_control, reward,
                                  done)  # obs_t, a_{t-1}, r_t, d_t
            else:
                print('Not recording...')
            text_embed = next_text_embed
            obs = next_obs

            collect_count += 1
            if collect_count % args.max_episode_count == 0 or collided or done:
                collect_count = 0
                itr += 1
                if itr >= 1:
                    model.train()
                    model = algo.optimize_agent(model, replay_buffer, itr - 3)
                    model.eval()
                    beta *= beta_decay
                    expert_flag = True if random.random() < beta else False
                    explore_flag = True if random.random() < 0.0 else False
                while True:
                    obs, _, done = env.reset()
                    print('Not recording')
                    if not done:
                        break
                algo.writer.add_scalar('Loss/task_reward', task_reward / (1e-6 + episode_count), itr)

                task_reward = 0
                episode_count = 0

                curr_waypoint = env.map.get_waypoint(agent.vehicle.get_location())
                destination = curr_waypoint.next(100.)[0].transform.location
                agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

    finally:
        if env is not None:
            env.destroy()

        pygame.quit()


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
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm-port',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--render',
        default=False,
        type=bool,
        help='Render display of 3rd person view (default: True)')
    argparser.add_argument(
        '--render_fog',
        default=False,
        type=bool,
        help='Render the fog in the environment (default: True)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',  # '1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--beta',
        help='The beta value for DAgger',
        default=1.0,
        type=float)
    argparser.add_argument(
        '--beta_decay',
        help='DAgger beta decay',
        default=0.5,
        type=float)
    argparser.add_argument(
        '--max_episode_count',
        help='Max count for each data collection phase',
        default=250,#150,
        type=int)
    argparser.add_argument(
        '--device',
        help='GPU device',
        default=1,
        type=int)
    argparser.add_argument(
        '--gpt_device',
        help='GPU device for gpt',
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
