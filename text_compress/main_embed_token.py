from obs_text_ae import AutoEncodersUni
from text_dataset import EmbedTextDataset
import torch
from pytorch_transformers import GPT2LMHeadModel
from typing import Iterable
from torch.nn import Module


def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    max_epochs = 20
    gpt_embed_size = 768
    device = 'cuda:0'
    model_save_path = './model_save/'
    data_save_path = './data_save/'

    training_dataset = EmbedTextDataset()
    criterion_lm = torch.nn.CrossEntropyLoss()

    model = AutoEncodersUni(latent_size=19, hidden_size=64, layers=3)
    model = torch.load(f'{model_save_path}autoencoder.pt')
    model.set_device(device)
    train_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=50, shuffle=True, num_workers=4)

    for i in range(6):
        gpt_lm_model = torch.load(f'{model_save_path}gpt_lm_model_{i}.pt')
        optimizer = torch.optim.Adam(get_parameters([gpt_lm_model]), lr=0.001)

        for epoch in range(max_epochs):
            avg_loss = 0
            count = 0
            for data in train_data_loader:
                text_embed = data['text_embed'].to(device)
                text_token = data['text_token'].to(device)

                batch_size = text_embed.shape[0]

                optimizer.zero_grad()
                loss_lm = 0.0
                gpt_embed = model.decoder_text(text_embed[:, i]).mean
                prediction = gpt_lm_model(gpt_embed.reshape(-1, gpt_embed_size).detach())

                loss_lm += criterion_lm(prediction,
                                        text_token[:, i].flatten(start_dim=0, end_dim=-1).detach())

                loss_lm.backward()
                optimizer.step()
                avg_loss += loss_lm.item()
                count += 1
            torch.save(gpt_lm_model, f'{model_save_path}gpt_lm_model_{i}.pt')
            print(f'{epoch}: loss:{avg_loss / count}')

