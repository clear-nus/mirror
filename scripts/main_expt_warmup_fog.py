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

from utils.audio_player import play

from envs.expt.carla_env import CarlaEnv
from envs.steering_wheel_control import SteeringWheelControl
from models.comm_cat_model import BisimRLModel
from models.cat_model import BisimModel

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

        env = CarlaEnv(render_display=args.render, render_fog=True, expt_stage='warmup-fog',
                       host=args.host, port=args.port,
                       tm_port=args.tm_port, frame_skip=1,
                       image_resolution=(args.width, args.height),
                       robot_model=robot_model,
                       mirror_model=mirror_model,
                       steering_agent=steering_control,
                       forward_record_flag=False)

        # Warm up 2 rounds with fog
        env.episode = 0
        env.total_episodes = 3
        while True:
            obs, _, done, lidar_obs = env.reset()
            if not done:
                break
        # Wait for red button to be pressed at the start
        while True:
            if steering_control.parseKey() or steering_control.parseButton():
                env.ready_mode = False
                break
        while True:
            if steering_control != None:
                if steering_control.is_wheel:
                    # Physical action using the steering wheel
                    control = steering_control.parseVehicleWheel()
                else:
                    # Physical action using the keyboard
                    control = steering_control.parseVehicleKey(env.clock.get_time())
                control = torch.tensor([control.throttle, control.steer])
            for i in range(3):
                next_obs, lidar_data_id, lidar_data_vel, reward, collided, done = env.step(control)
            if collided:
                play(file='./utils/robot_comm_audio/collided.wav')

            if done or collided:  # If episode ended or collided
                env.episode += 1
                env.init_accel = False
                env.ready_mode = True
                if env.episode == 3:
                    break
                else:
                    while True:
                        obs, _, done, lidar_obs = env.reset()
                        if not done:
                            break
                    ready_flag = False
                    while not ready_flag:
                        if steering_control.parseButton():
                            ready_flag = True
                            env.ready_mode = False

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
        help='choose which communication agent to use (Our Method (OM)/ASE/Show Nothing (SN))',
        default='OM',
        type=str
    )
    argparser.add_argument('--id',
                           help='participant id',
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
