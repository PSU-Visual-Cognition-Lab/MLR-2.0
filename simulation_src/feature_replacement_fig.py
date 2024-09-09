from label_network import load_checkpoint_colorlabels, load_checkpoint_shapelabels, s_classes, vae_shape_labels, vae_color_labels,c_classes
import torch
from mVAE import vae, load_checkpoint, image_activations, activations
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import Dataset
import matplotlib.pyplot as plt
from joblib import dump, load
from torchvision import utils
import torch.nn.functional as F
import os

checkpoint_folder_path = f'feature_replacement/' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

v = '2' # which model version to use, set to '' for the most recent
load_checkpoint(f'output_mnist_2drecurr{v}/checkpoint_most_recent.pth')
#load_checkpoint_shapelabels(f'output_label_net{v}/checkpoint_shapelabels5.pth')
load_checkpoint_colorlabels(f'output_label_net/checkpoint_colorlabels1.pth')
#clf_shapeS=load(f'classifier_output{v}/ss.joblib')

colornames = ["red", "green", "blue", "purple", "yellow", "cyan", "orange", "brown", "pink", "white"]

mnist_transforms = {'retina':False, 'colorize':True}
mnist_dataset = Dataset('mnist', mnist_transforms)
mnist_loader = mnist_dataset.get_loader(10)

vae.eval()
with torch.no_grad():
    for label in range(10):
        data_iter = iter(mnist_loader)
        data = next(data_iter)
        data = data[0].view(10,3,28,28).cuda()
        
        
        # build one hot vectors to be passed to the label networks
        onehot_label = F.one_hot(torch.tensor([label]).cuda(), num_classes=c_classes).float().cuda() # shape

        # generate shape latents from the labels n = noise
        z_color_labels = vae_color_labels(onehot_label)
        output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,a,b = vae(data,whichdecode='shape')

        z_shape_img = vae.sampling(mu_shape,log_var_shape)

        # pass latents from label network through encoder
        recon_shape = vae.decoder_cropped(z_shape_img, z_color_labels, 0, 0)
        
        utils.save_image(torch.cat([data,recon_shape],dim=0),f'feature_replacement/replace_with_{colornames[label]}.png',nrow=10)