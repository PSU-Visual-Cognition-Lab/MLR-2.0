import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from sklearn.metrics import classification_report, confusion_matrix
from training_constants import training_components, text_to_tensor
import torch.nn.functional as F
import os

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy
from MLR_src.mVAE import batch_samples


bs = 100
s_classes = 62
c_classes = 10

def train_labelnet(dataloaders, vae, epoch_count, shape_z_dim, color_z_dim, object_z_dim, checkpoint_folder, trained_components):
    if not os.path.exists('training_samples/'):
        os.mkdir('training_samples/')
    
    if not os.path.exists(f'training_samples/{checkpoint_folder}/'):
        os.mkdir(f'training_samples/{checkpoint_folder}/')
    
    sample_folder_path = f'training_samples/{checkpoint_folder}/label_net_samples/'
    if not os.path.exists(sample_folder_path):
        os.mkdir(sample_folder_path)
    
    vae_shape_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=shape_z_dim)
    vae_object_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=object_z_dim)
    vae_color_labels= VAEcolorlabels(xlabel_dim=c_classes, hlabel_dim=7,  zlabel_dim=color_z_dim)

    optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
    optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())
    optimizer_objectlabels= optim.Adam(vae_object_labels.parameters())

    
    label_nets = {}
    if 'shape' in trained_components:
        label_nets['shape'] = [vae_shape_labels, optimizer_shapelabels]
    if 'object' in trained_components:
        label_nets['object'] = [vae_object_labels, optimizer_objectlabels]
    if 'color' in trained_components:
        label_nets['color'] = [vae_color_labels, optimizer_colorlabels]

    for whichcomponent in label_nets:
        label_net, optimizer = label_nets[whichcomponent]
        for epoch in range (1,epoch_count):
            #train_labels(vae, label_net, whichcomponent, epoch, train_loader, optimizer, folder_path):
            whichloaders =  training_components[whichcomponent][0]
            train_labels(vae, label_net, whichcomponent, epoch, whichloaders, dataloaders, optimizer, sample_folder_path)
            
    checkpoint =  {
            'state_dict_shape_labels': vae_shape_labels.state_dict(),
            'state_dict_color_labels': vae_color_labels.state_dict(),
            'state_dict_object_labels': vae_object_labels.state_dict(),
            'optimizer_shape' : optimizer_shapelabels.state_dict(),
            'optimizer_color': optimizer_colorlabels.state_dict(),
            'optimizer_object': optimizer_objectlabels.state_dict(),

            'shape_z_dim': shape_z_dim,
            'color_z_dim': color_z_dim,
            'object_z_dim': object_z_dim
                }
    torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/label_network_checkpoint.pth')

# shape label network
class VAEshapelabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAEshapelabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape


    def sampling_labels (self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)
        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return  z_shape_label

# color label network
class VAEcolorlabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, zlabel_dim):
        super(VAEcolorlabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label = nn.Linear(hlabel_dim, zlabel_dim)  # mu color
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim)  # log-var color

    def sampling_labels(self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n=1):
        device = next(self.parameters()).device
        x_labels.to(device)
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label = self.fc22label(h)
        z_color_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return  z_color_label

def load_checkpoint_labels(filepath, label_type, d=0):
    assert label_type in ["shape", "color", "object"], f"label_type: {label_type} is invalid, must be one of: shape, color, object"
    print(f"loading {label_type} label network")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
    else:
        device = 'cpu'

    checkpoint = torch.load(filepath)
    if f'{label_type}_z_dim' not in checkpoint:
        z_dim = 8
    else:
        z_dim = checkpoint[f'{label_type}_z_dim']

    label_net_dict = {"shape":VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=z_dim), 
                      "object": VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=z_dim),
                      "color": VAEcolorlabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=z_dim)}
    
    label_net = label_net_dict[label_type]
    label_net.load_state_dict(checkpoint[f'state_dict_{label_type}_labels'])
    for parameter in label_net.parameters():
        parameter.requires_grad = False
    label_net.eval().to(device)
    return label_net

def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')
    e=criterion(label_act,image_act)

    return e

def train_labels(vae, label_net, whichcomponent, epoch, whichloaders, train_loader, optimizer, folder_path):    
    device = next(vae.parameters()).device
    vae.eval()
    train_loss = 0

    label_net.train().to(device)

    dataiter = train_loader

    max_iter = 100
    for i, _  in enumerate(range(max_iter+1)):
        optimizer.zero_grad()

        #image, labels = next(dataiter)
        image, labels = batch_samples(whichloaders, train_loader, whichcomponent, False)
        image.to(device)
        labels_for_shape=labels[0].clone() # shape or object, depending on dataset
        labels_for_color=labels[1].clone()
              
        #image = image[1].cuda() # dataset must be retinal for this(?)
        batch_size = image.size(0)

        if whichcomponent == 'color':
            labels_color = labels_for_color  # get the color label from the dataloader
            labels_color = labels_color.to(device)
            color_oneHot = F.one_hot(labels_color, num_classes=10)
            color_oneHot = color_oneHot.float()
            input_one_hot = color_oneHot.to(device)

        else:
            labels_shape = labels_for_shape.to(device)  #get the shape label from the dataloader
            input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 36 classes in emnist, 10 classes in f-mnist
            input_oneHot = input_oneHot.float()
            input_one_hot = input_oneHot.to(device)
        
        n = 1 # sampling noise
        z_label = label_net(input_one_hot,n)  #run a label through the model to generate a latent representation
        

        activations = vae.activations(image, False)  #load activity for each of the latent spaces based on the image
        z_actual = activations[whichcomponent]   #extract the activity for that latent representation
        '''
        # don't use the randomized output for training, instead just use the mu
        
        mu_shape, _, mu_color, _, hskip = vae.encoder(image)
        mu_object, _ = vae.encoder_object(image)
        if whichcomponent == 'shape':
            z_actual = mu_shape
        elif whichcomponent == 'color':
            z_actual = mu_color
        elif whichcomponent == 'object':
            z_actual = mu_object
        '''

        # train shape label net
        
        #loss_of_labels = loss_label(z_label, z_actual)   #compute the error

        if whichcomponent == 'object':
            recon_from_label = vae.decoder_object(z_label).to(device)
            # compare against grayscale target (same as loss_function_shape does)
            x_gray = image.view(-1, 3, 28, 28).mean(1)
            x_gray = torch.stack([x_gray, x_gray, x_gray], dim=1).to(device)
            loss_of_labels = F.binary_cross_entropy(recon_from_label.view(-1, 28*28*3), x_gray.view(-1, 28*28*3), reduction='sum')
        else:
            loss_of_labels = loss_label(z_label, z_actual)

        loss_of_labels.backward(retain_graph = True)    #adjust the weights so that the label network output resembles the latent activation
        train_loss += loss_of_labels.item()      

        optimizer.step()

        if i % max_iter == 0 and i > 0:   #every 100 samples grab a random sample
            label_net.eval()
            vae.eval()

            if whichcomponent == 'color':   #grab the appropriate decoder from the VAE
                feature_decoder = vae.color_decode_wrapper
            elif whichcomponent == 'shape':
                feature_decoder = vae.decoder_shape
            else:    
                feature_decoder = vae.decoder_object

            with torch.no_grad():   #Use them to reconstruct an object
                #feature_recon = feature_decoder(z_actual)         # recon from an actual object
                if whichcomponent == 'color':
                    feature_recon, _, _, _, _,_,_ = vae(image, 'color', ['color'])
                elif whichcomponent == 'shape':
                    feature_recon, _, _, _, _,_,_= vae(image, 'cropped', ['shape'])
                else:
                    feature_recon, _, _, _, _,_,_= vae(image, 'cropped_object', ['object'])
                feature_recon_label = feature_decoder(z_label)


                sample_size = batch_size if batch_size < 30 else 30
                orig_imgs = image[:sample_size].to(device)
                feature_recon = feature_recon[:sample_size].to(device)
                feature_recon_label = feature_recon_label[:sample_size].to(device)

            output_img = torch.cat(
                    [orig_imgs,
                     feature_recon.view(sample_size, 3, 28, 28),
                     feature_recon_label.view(sample_size, 3, 28, 28)
                     ], 0)
            rows = 3
            #print(type(output_img))
            #this next bit collapses the long image into a stack of rows so that the text can be added
            #convert the sample_size*rows x 3 x 28 x 28 tensor into a  stack that is now 3 x rows*28 x sample_size*28
            output_img2 = output_img.view(rows,sample_size,3,28,28)
            output_img2 = output_img2.permute(0,2,3,1,4).contiguous().view(rows,3,28,sample_size*28)
            output_img2 = output_img2.permute(1,0,2,3).contiguous().view(3,rows*28,sample_size*28)

            channels, height, width = output_img2.shape
            header_height = 20            
            # Create new tensor with extra height for text
            new_height = height + header_height
            new_tensor = torch.ones(channels, new_height, width) * 0.8  # Light gray background
            new_tensor[:, header_height:, :] = output_img2
            text_tensor = text_to_tensor("Image / recon from encoder / recon from label ",header_height,width)
            new_tensor[:, :header_height, :] = text_tensor
            utils.save_image(new_tensor,
                f'{folder_path}{whichcomponent}{str(epoch).zfill(5)}_{str(i).zfill(5)}.png',
                nrow = 1,
                normalize=False,
            )

        if i > max_iter + 1:
            break
    print(f'====> Epoch: {epoch} Average loss {whichcomponent}: {train_loss}')