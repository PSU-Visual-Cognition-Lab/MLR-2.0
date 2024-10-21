from MLR_src.label_network import *
import torch

def train_labelnet(dataloaders, vae, epoch_count, checkpoint_folder):
    optimizer = optim.Adam(vae.parameters())

    optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
    optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())
    for epoch in range (1,epoch_count):
        train_labels(vae, epoch, dataloaders[0], optimizer_shapelabels, optimizer_colorlabels)
        
    checkpoint =  {
            'state_dict_shape_labels': vae_shape_labels.state_dict(),
            'state_dict_color_labels': vae_color_labels.state_dict(),

            'optimizer_shape' : optimizer_shapelabels.state_dict(),
            'optimizer_color': optimizer_colorlabels.state_dict(),

                }
    torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/label_network_checkpoint.pth')