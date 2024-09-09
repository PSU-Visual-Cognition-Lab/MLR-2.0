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

mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', 2, None, False, None, False, True)
mnist_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=2, shuffle=True,  drop_last= True)
vae.eval()
with torch.no_grad():
    label = 35
    print(vals[label])
    data_iter = iter(mnist_loader)
    
    target = 0
    while target != 7:
        data = data_iter.next()
        img = data[0][0].cuda()
        target = data[1][0][0].item()
        img = img.view(1,3,28,28)

    # build one hot vectors to be passed to the label networks
    onehot_label = F.one_hot(torch.tensor([label]).cuda(), num_classes=s_classes).float().cuda() # shape

    # generate shape latents from the labels n = noise
    z_shape_labels = vae_shape_labels(onehot_label, n = 10)
    output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(img,whichdecode='shape')

    z_shape_img = vae.sampling(mu_shape,log_var_shape)
    z_color_img = vae.sampling(mu_color,log_var_color)

    combined_z_shape = (1.5*z_shape_labels+z_shape_img)*(1/2)
    print(combined_z_shape.size())

    # pass latents from label network through encoder
    recon_shape = vae.decoder_cropped(combined_z_shape, z_color_img, 0, 0)
    
    '''    #comb_img = torch.cat([comb_img, comb_img],0)
    l1,l2,z_shape, z_color, z_location = activations(comb_img)

    pred_ss = clf_shapeS.predict(z_shape.cpu())
    pred_proba = clf_shapeS.predict_proba(z_shape.cpu())

    recon_shape = vae.decoder_shape(z_shape, 0, 0)'''

    utils.save_image(recon_shape,'percept_concept_8B.png')
    utils.save_image(img,'percept_8.png')

    '''    print(pred_ss)
    print(vals[pred_ss[0].item()])'''