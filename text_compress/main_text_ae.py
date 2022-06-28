from obs_text_ae import AutoEncoders
from text_dataset import ObsTextDataset
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    max_epochs = 200
    device = 'cuda:0'
    model_save_path = './model_save/'
    data_save_path = './data_save/'

    training_dataset = ObsTextDataset()

    h_direction_list = training_dataset.h_direction_list
    v_direction_list = training_dataset.v_direction_list

    model = AutoEncoders(h_direction_list=h_direction_list, v_direction_list=v_direction_list,
                         latent_size=10, hidden_size=48, layers=3)
    # model.load_models(model_save_path)
    model.set_device(device)

    training_dataset = ObsTextDataset()
    training_dataset.load_training_data(data_save_path)

    train_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=50, shuffle=True, num_workers=4)

    for h_direction in h_direction_list:
        for v_direction in v_direction_list:
            training_dataset.set_sample_direction(h_direction, v_direction)
            optimizer = torch.optim.Adam(model.autoencoders[f'{h_direction}-{v_direction}'].parameters(), lr=0.001)
            for epoch in range(max_epochs):
                avg_rec = 0.0
                count = 0
                for data in train_data_loader:
                    optimizer.zero_grad()

                    lidar_dis = data['lidar_dis'].to(device)
                    gpt_embed = data['gpt_embed'].to(device)
                    text_token = data['text_token'].to(device)
                    speed = data['speed'].to(device)

                    autoencoder = model.autoencoders[f'{h_direction}-{v_direction}']

                    latent_state_dist = autoencoder.encoder(torch.cat([lidar_dis, speed], dim=-1))
                    latent_state_rsample = latent_state_dist.rsample()

                    gpt_embed_dist = autoencoder.decoder_text(latent_state_rsample)

                    # rec_loss = -gpt_embed_dist.log_prob(gpt_embed).mean()
                    rec_loss = (gpt_embed - gpt_embed_dist.mean).pow(2).sum(dim=-1).mean()
                    kl_loss = 0.01 * latent_state_rsample.pow(2).mean()

                    loss = rec_loss + kl_loss

                    avg_rec += rec_loss
                    count += 1

                    loss.backward()
                    optimizer.step()

                    torch.save(autoencoder, f'{model_save_path}autoencoder_{h_direction}-{v_direction}.pt')
                print(f'{epoch}: rec_loss:{avg_rec.item()/count}, kl_loss:{kl_loss}')
    torch.save(model, model_save_path + 'autoencoders.pt')
