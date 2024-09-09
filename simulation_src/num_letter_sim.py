from label_network import load_checkpoint_colorlabels, load_checkpoint_shapelabels, s_classes, vae_shape_labels
import torch
from mVAE import vae, load_checkpoint, image_activations, activations
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder
import matplotlib.pyplot as plt
from joblib import dump, load
from torchvision import utils
import torch.nn.functional as F

v = '' # which model version to use, set to '' for the most recent
load_checkpoint(f'output_emnist_recurr{v}/checkpoint_300.pth')
load_checkpoint_shapelabels(f'output_label_net{v}/checkpoint_shapelabels5.pth')
#load_checkpoint_colorlabels(f'output_label_net{v}/checkpoint_colorlabels10.pth')
clf_shapeS=load(f'classifier_output{v}/ss.joblib')

vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# char combinations: [1,3: B], [L,1: U], []

vae.eval()
with torch.no_grad():
    num1 = 21#1#1#1#21#5#0#1#1#1#1#1 #1 # first char 21
    num2 = 1#4#7#3#15#6#1#7#8#9#2#4 #3 # second char 1
    x1, x2 = 0, 8 # locations for each img
    #colors = torch.randint(low=0, high=10, size=(2,))

    # build one hot vectors to be passed to the label networks
    num_labels = F.one_hot(torch.tensor([num1, num2]).cuda(), num_classes=s_classes).float().cuda() # shape
    loc_labels = torch.zeros((2,100)).cuda()
    loc_labels[0][x1], loc_labels[1][x2] = 1, 1 # location
    #col_labels = F.one_hot(torch.tensor([num1, num2]).cuda(), num_classes=10).float().cuda()

    # generate shape latents from the labels n = noise
    z_shape_labels = vae_shape_labels(num_labels, n = 10)

    # location latent from the location vector
    z_location = vae.location_encoder(loc_labels)

    # pass latents from label network through encoder
    recon_retinal = vae.decoder_retinal(z_shape_labels, 0, z_location, None, 'shape')
    # clamp shape recons to form one image of the combined numbers
    img1 = recon_retinal[0,:,:,6:34]
    img2 = recon_retinal[1,:,:,6:34]
    #comb_img = torch.log((img1*255)+(img2*255)) * (1/9)
    comb_img = torch.clamp(img1+img2, 0, 0.5) *1.5
    comb_img = comb_img.view(1,3,28,28)

    
    #comb_img = torch.cat([comb_img, comb_img],0)
    l1,l2,z_shape, z_color, z_location = activations(comb_img)

    pred_ss = clf_shapeS.predict(z_shape.cpu())
    pred_proba = clf_shapeS.predict_proba(z_shape.cpu())

    recon_shape = vae.decoder_shape(z_shape, 0, 0)
    utils.save_image(comb_img,f'{vals[num1]}_{vals[num2]}_pred_{vals[pred_ss[0].item()]}3.png')
    utils.save_image(recon_shape,f'{vals[num1]}_{vals[num2]}_sim_recon3.png')
    utils.save_image(img1,f'{vals[num1]}_img3.png')
    utils.save_image(img2,f'{vals[num2]}_img3.png')
    print(pred_ss)
    print(vals[pred_ss[0].item()])