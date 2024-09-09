import torch
from mVAE import vae, load_checkpoint
from torchvision import utils
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder
from random import random

v = '' # which model version to use, set to '' for the most recent
load_checkpoint(f'output_emnist_recurr{v}/checkpoint_300.pth')


vae.eval()
b = 100
grid_size = int(b**(1/2))
with torch.no_grad():
    # generate latent shape samples:
    #sample = torch.rand(b,128).cuda()
    mu_shape = torch.cat([torch.rand(b//2,16).cuda(),torch.rand(b//2,16).cuda()],dim=0) *10 + (torch.randn(b,16).cuda()) #* 1.2# + 0.1 #((torch.rand(b,16).cuda()*0.6) + (torch.randn(b,16).cuda()*0.4)) * 2 #+ 0.1#vae.fc31(sample) * 1.2 #)
    
    #print(mu_shape[0])
    log_var_shape = torch.rand(b,16).cuda()* 0.8 -4 #4# ((torch.rand(b,16).cuda()*0.2)+4) #vae.fc32(sample) #()+ (torch.randn(b,16).cuda()*0.4)
    #print(log_var_shape[0])
    z_img = vae.sampling(mu_shape,log_var_shape)
    #z_img  = torch.randn(b,16).cuda()

    # run tsne on sampling:
    tensors = [item.cpu().detach().numpy() for item in z_img]
    tensors_array = np.array(tensors)
    tsne = TSNE(n_components=2, perplexity=30)
    embedded_data = tsne.fit_transform(tensors_array)

    # assign samples to grid locations based on tsne embeddings:
    position_list = embedded_data.tolist()
    for i in range(len(position_list)):
        position_list[i] += [ i]

    sorted_by_y = sorted(position_list, key=lambda x: x[1], reverse=True)
    row_list = []
    for i in range(grid_size):
        t_row = sorted_by_y[:grid_size]
        t_row = sorted(t_row, key=lambda x: x[0], reverse=True)
        row_list += [t_row]
        sorted_by_y = sorted_by_y[grid_size:]
    
    l_values = [item[2] for sublist in row_list for item in sublist]
    sorted_z = torch.zeros(b,16).cuda()
    for i in range(b):
        j = l_values[i]
        sorted_z[i]  = z_img[j]

    recon_sample_tsne = vae.decoder_shape(sorted_z, 0, 0)
    recon_sample = vae.decoder_shape(z_img, 0, 0)
    utils.save_image(recon_sample_tsne,f'random_shape_sampling{b}.png',grid_size)
    if b <= 200:
        utils.save_image(recon_sample,f'random_shape_sampling.png')