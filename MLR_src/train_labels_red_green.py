from label_network import *
import torch
from mVAE import vae, load_checkpoint
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder

load_checkpoint('output_red_green/checkpoint_150.pth')
bs = 50
#load_checkpoint_shapelabels('output_label_net/checkpoint_shapelabels5.pth')


# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
#emnist_dataset, emnist_skip, emnist_test_dataset = dataset_builder('emnist', bs, None, False, None, False, True)
#mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', bs, None, False, None, False, True)
mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', bs, {0:[0,1,2,3,4],1:[5,6,7,8,9]}, False, None, False, True)
opp_mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', bs, {1:[0,1,2,3,4],0:[5,6,7,8,9]}, False, None, False, True)
#concat datasets and init dataloaders
train_loader_noSkip = torch.utils.data.DataLoader(dataset= mnist_dataset, batch_size=bs, shuffle=True,  drop_last= True)
test_loader = torch.utils.data.DataLoader(dataset= opp_mnist_dataset, batch_size=bs, shuffle=True,  drop_last= True)

for epoch in range (1,21):
   
    train_labels(epoch, train_loader_noSkip)
    test_opposite_colors(test_loader,n=1)
        
    if epoch in [1,5,10,20]:
        checkpoint =  {
                 'state_dict_shape_labels': vae_shape_labels.state_dict(),
                 'state_dict_color_labels': vae_color_labels.state_dict(),

                 'optimizer_shape' : optimizer_shapelabels.state_dict(),
                 'optimizer_color': optimizer_colorlabels.state_dict(),

                      }
        torch.save(checkpoint,f'output_label_net_red_green/checkpoint_shapelabels'+str(epoch)+'.pth')
        torch.save(checkpoint, f'output_label_net_red_green/checkpoint_colorlabels' + str(epoch) + '.pth')

