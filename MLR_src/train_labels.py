from label_network import *
import torch
from mVAE import vae, load_checkpoint
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import Dataset
import os
v=2

checkpoint_folder_path = f'output_label_net/' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

load_checkpoint(f'output_mnist_2drecurr{v}/checkpoint_most_recent.pth')
bs = 50
#load_checkpoint_shapelabels('output_label_net/checkpoint_shapelabels5.pth')

transforms = {'colorize':True}

#emnist_dataset = Dataset('emnist', transforms)
mnist_dataset = Dataset('mnist', transforms)
mnist_loader = mnist_dataset.get_loader(bs)
#train_loader_noSkip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_dataset, mnist_dataset, mnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)
for epoch in range (1,21):
   
    train_labels(epoch, mnist_loader)
        
    if epoch in [1,5,10,20]:
        checkpoint =  {
                 'state_dict_shape_labels': vae_shape_labels.state_dict(),
                 'state_dict_color_labels': vae_color_labels.state_dict(),

                 'optimizer_shape' : optimizer_shapelabels.state_dict(),
                 'optimizer_color': optimizer_colorlabels.state_dict(),

                      }
        torch.save(checkpoint,f'output_label_net/checkpoint_shapelabels'+str(epoch)+'.pth')
        torch.save(checkpoint, f'output_label_net/checkpoint_colorlabels' + str(epoch) + '.pth')

