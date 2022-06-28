from torch.utils.data import Dataset
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

class EmbedTextDataset(Dataset):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_device = 'cuda:1'

        path = './data_save'
        text_embed_np = np.concatenate([#np.load(f'{path}/gpt_text_embed_0.npy'),
                                        # np.load(f'{path}/gpt_text_embed_1.npy'),
                                        np.load(f'{path}/gpt_text_embed_2.npy')], axis=0)
        text_np = np.concatenate([#np.load(f'{path}/text_0.npy'),
                                  # np.load(f'{path}/text_1.npy'),
                                  np.load(f'{path}/text_2.npy')], axis=0)

        text_embed_np = np.stack([text_embed_np[:, 0*20:1*20-1],
                                  text_embed_np[:, 1*20:2*20-1],
                                  text_embed_np[:, 2*20:3*20-1],
                                  text_embed_np[:, 3*20:4*20-1],
                                  text_embed_np[:, 4*20:5*20-1],
                                  text_embed_np[:, 5*20:6*20-1]], axis=1) * 15.0
        self.data_size = text_np.shape[0]

        # generate text_token from text np
        text_token_list = []
        for i in range(self.data_size):
            sub_text_token_list = []
            for j in range(6):
                text = text_np[i, j]
                text_token = self.tokenizer.encode(text)
                text_token = torch.tensor(text_token).to(self.gpt_device)
                text_length = text_token.size(0)
                max_length = 13
                if text_length <= max_length:
                    token_pad = torch.zeros(max_length - text_length).type(torch.int64).to(text_token.device)
                    text_token = torch.cat((text_token, token_pad))
                sub_text_token_list += [text_token]
            text_token_list += [torch.stack(sub_text_token_list, dim=0)]
        self.text_token_torch = torch.stack(text_token_list, dim=0).detach().cpu()
        self.text_embed_torch = torch.as_tensor(text_embed_np).detach().cpu()

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        return {'text_embed': self.text_embed_torch[i],
                'text_token': self.text_token_torch[i]}


class ObsTextDataset(Dataset):
    def __init__(self):
        self.none_level = {'front': 0.8, 'rear': 0.8, 'left-front': 0.8, 'right-front': 0.8, 'left-rear': 0.8,
                           'right-rear': 0.8}
        self.h_direction_list = ['front', 'rear', 'left-front', 'right-front', 'left-rear', 'right-rear']

        self.v_direction_list = ['front', 'rear']
        self.speed_list = ['fast', 'slow']

        self.gpt_device = 'cuda:0'
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.gpt_device)
        self.gpt_model.eval()

        self.sample_direction = 'front-front'

        self.lidar_dis_dict = {}
        self.gpt_embed_dict = {}
        self.text_token_dict = {}
        self.speed_dict = {}

    def set_sample_direction(self, h_direction, v_direction):
        self.sample_direction = f'{h_direction}-{v_direction}'

    def load_training_data(self, save_path='./data_save'):
        for h_direction in self.h_direction_list:
            for v_direction in self.v_direction_list:
                data_dict = np.load(f'{save_path}{h_direction}-{v_direction}.npz')
                self.lidar_dis_dict[f'{h_direction}-{v_direction}'] = torch.tensor(data_dict['lidar_dis']).type(
                    torch.float)
                self.gpt_embed_dict[f'{h_direction}-{v_direction}'] = torch.tensor(data_dict['gpt_embed']).type(
                    torch.float)
                self.text_token_dict[f'{h_direction}-{v_direction}'] = torch.tensor(data_dict['text_token']).type(
                    torch.float)
                self.speed_dict[f'{h_direction}-{v_direction}'] = torch.tensor(data_dict['speed']).type(torch.float)

    def generate_training_data(self, resolution, save_path='./data_save/'):
        for h_direction in self.h_direction_list:
            for v_direction in self.v_direction_list:
                lidar_dis_np, gpt_embed_np, text_token_np, speed_np = self.generate_direction_data(h_direction,
                                                                                                   v_direction,
                                                                                                   resolution,
                                                                                                   save_path)
                np.savez(f'{save_path}{h_direction}-{v_direction}',
                         lidar_dis=lidar_dis_np, gpt_embed=gpt_embed_np, text_token=text_token_np, speed=speed_np)

    def generate_direction_data(self, h_direction, v_direction, resolution, save_path='./data_save/'):
        size = int(1.0 / resolution)

        lidar_dis_list = []
        gpt_embed_list = []
        text_token_list = []
        speed_list = []
        for i in range(size):
            lidar_dis = 0.0 + i * resolution
            if lidar_dis > 1.0:
                lidar_dis = 1.0

            for speed in self.speed_list:
                gpt_embed, text_token = self.generate_text(lidar_dis, h_direction, v_direction, speed,
                                                           self.none_level[h_direction])

                if speed == 'fast':
                    speed_obs = 1
                else:
                    speed_obs = 0
            lidar_dis_list += [np.array([lidar_dis])]
            gpt_embed_list += [gpt_embed]
            text_token_list += [text_token]
            speed_list += [np.array([speed_obs])]

        lidar_dis_np = np.stack(lidar_dis_list, axis=0)
        gpt_embed_np = np.stack(gpt_embed_list, axis=0)
        text_token_np = np.stack(text_token_list, axis=0)
        speed_np = np.stack(speed_list, axis=0)

        return lidar_dis_np, gpt_embed_np, text_token_np, speed_np

    def generate_text(self, lidar_dis, h_direction, v_direction, speed, none_level=1.0):
        # give beam reading, return generated gpt-text-embed
        if lidar_dis < none_level:
            if v_direction == 'rear':
                if speed == 'fast':
                    text = f"Car is approaching fast from your {h_direction}"
                else:
                    text = f"Car is moving slowly at your {h_direction}"
            else:
                if speed == 'fast':
                    text = f"Car is moving fast at your {h_direction}"
                else:
                    text = f"Car is slowing down at your {h_direction}"
        elif none_level <= lidar_dis:
            text = f"No car detected in the nearby range at your {h_direction}"
        text_token = self.tokenizer.encode(text)
        text_token = torch.tensor(text_token).to(self.gpt_device)
        text_length = text_token.size(0)
        max_length = 13
        if text_length <= max_length:
            token_pad = torch.zeros(max_length - text_length).type(torch.int64).to(text_token.device)
            text_token = torch.cat((text_token, token_pad))

        gpt_embed = self.gpt_model.transformer(text_token.unsqueeze(dim=0).to(self.gpt_device))[0]
        gpt_embed = torch.flatten(gpt_embed, start_dim=-2, end_dim=-1)[0].detach().cpu().numpy()
        text_token = text_token.detach().cpu().numpy()

        print(speed, h_direction, v_direction, text_length, text_token.shape)

        # pad array different length
        return gpt_embed, text_token

    def __len__(self):
        return self.lidar_dis_dict[self.sample_direction].size(0)

    def __getitem__(self, i):
        return {'lidar_dis': self.lidar_dis_dict[self.sample_direction][i],
                'gpt_embed': self.gpt_embed_dict[self.sample_direction][i],
                'text_token': self.text_token_dict[self.sample_direction][i],
                'speed': self.speed_dict[self.sample_direction][i]}


if __name__ == '__main__':
    dataset = ObsTextDataset()

    dataset.generate_training_data(0.02)
