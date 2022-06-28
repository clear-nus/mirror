import numpy as np


class DataRecorder:
    def __init__(self, path):
        self.path = path
        # allocate memory for buffers
        self.task_rewards = []
        self.speeds = []
        self.global_speeds = []
        self.collisions = []
        self.car_poses = []

        self.human_actions = []
        self.comm_lidar_actions = []
        self.comm_text_actions = []

        self.lidar_car_ids = []
        self.speeches = []

        self.rear_spawn_pos = []
        self.rear_speed = []
        self.front_spawn_pos = []
        self.front_speed = []
        self.episode_num = []

        # add some raw data buffers

    def add_case_setup(self,front_spawn_pos, rear_spawn_pos, rear_speed, front_speed, episode_num):
        self.front_spawn_pos += [front_spawn_pos]
        self.rear_spawn_pos += [rear_spawn_pos]
        self.episode_num += [episode_num]
        self.front_speed += [front_speed]
        self.rear_speed += [rear_speed]

    def add_data(self, task_reward,
                 comm_lidar_action,
                 comm_text_action,
                 speed, collision,
                 human_action,
                 global_speed,
                 car_pos,
                 lidar_car_id,
                 speech):
        # add traj data to buffer
        if task_reward is not None:
            self.task_rewards += [task_reward]
        if speed is not None:
            self.speeds += [speed]
        if collision is not None:
            self.collisions += [collision]
        if global_speed is not None:
            self.global_speeds += [global_speed]
        if car_pos is not None:
            self.car_poses += [car_pos]

        if comm_lidar_action is not None:
            self.comm_lidar_actions += [comm_lidar_action]
        if comm_text_action is not None:
            self.comm_text_actions += [comm_text_action]
        if human_action is not None:
            self.human_actions += [human_action]

        if lidar_car_id is not None:
            self.lidar_car_ids += [lidar_car_id]

        if speech is not None:
            self.speeches += [speech]

    def save_data(self, tech: str, case: str):
        # save traj of each episode
        front_spawn_pos_np = np.array(self.front_spawn_pos)
        rear_spawn_pos_np = np.array(self.rear_spawn_pos)
        episode_num_np = np.array(self.episode_num)
        rear_speed_np = np.array(self.rear_speed)
        front_speed_np = np.array(self.front_speed)

        task_rewards_np = np.array(self.task_rewards)
        speeds_np = np.array(self.speeds, dtype='<U55')
        collisions_np = np.array(self.collisions)
        global_speeds_np = np.array(self.global_speeds)
        car_poses_np = np.array(self.car_poses)

        human_actions_np = np.array(self.human_actions)
        comm_lidar_actions_np = np.array(self.comm_lidar_actions)
        comm_text_actions_np = np.array(self.comm_text_actions)

        lidar_car_ids_np = np.array(self.lidar_car_ids)
        speeches_np = np.array(self.speeches)

        file_name = f'traj_{tech}_{case}.npz'
        np.savez(f'{self.path}/{file_name}',
                 front_speed=front_speed_np,
                 rear_speed=rear_speed_np,
                 front_spawn_pos=front_spawn_pos_np,
                 rear_spawn_pos=rear_spawn_pos_np,
                 episode_num=episode_num_np,
                 task_reward=task_rewards_np,
                 speed=speeds_np,
                 collisions=collisions_np,
                 car_poses=car_poses_np,
                 global_speeds=global_speeds_np,
                 human_actions=human_actions_np,
                 comm_lidar_action=comm_lidar_actions_np,
                 comm_text_action=comm_text_actions_np,
                 lidar_car_ids=lidar_car_ids_np,
                 speeches=speeches_np)

        print(f'Save file {file_name} to {self.path}')

    def clear_buffer(self):
        self.front_spawn_pos.clear()
        self.rear_spawn_pos.clear()
        self.front_speed.clear()
        self.rear_speed.clear()
        self.episode_num.clear()

        self.task_rewards.clear()
        self.speeds.clear()
        self.collisions.clear()
        self.global_speeds.clear()
        self.car_poses.clear()

        self.comm_lidar_actions.clear()
        self.comm_text_actions.clear()
        self.human_actions.clear()

        self.lidar_car_ids.clear()
        self.speeches.clear()
