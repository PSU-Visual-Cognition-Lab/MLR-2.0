#MLR 2.0

#The second installment of the MLR model line, written largely by Ian Deal and Brad Wyble
#This original version of this model is published in
#Hedayati, S., O’Donnell, R. E., & Wyble, B. (2022). A model of working memory for latent representations. Nature Human Behaviour, 6(5), 709-719.
#And the code in that work is a variant of
# MNIST VAE from http://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Modified by Brad Wyble, Shekoo Hedayati

#In this version, the model adds to the original MLR model the following features:
#-a large Retina  (100 pixels wide)
#-Convolutional encoder and decoder
#-Location latent space  (in the horizontal diection)
#-improved loss functions for shape and color
#-White is now one of the 10 colors
#-Skip connection trained on bi-color stimuli
#-Label networks  akin to SVRHM paper:
#Hedayati, S., Beaty, R., & Wyble, B. (2021). Seeking the Building Blocks of Visual Imagery and Creativity in a Cognitively Inspired Neural Network. arXiv preprint arXiv:2112.06832.



# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os
from torch.utils.data import DataLoader, Subset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

#torch.set_default_dtype(torch.float64)

# load a saved vae checkpoint
def load_checkpoint(filepath, d=0):
    vae, z = vae_builder()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
    else:
        device = 'cpu'
    checkpoint = torch.load(filepath,device, weights_only = True)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.to(device)
    return vae

global colorlabels
numcolors = 0

colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "white"]
colorlabels = np.random.randint(0, 10, 1000000)
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [1-colorrange,1-colorrange,1-colorrange]
]


#comment this
def Colorize_func(img):
    global numcolors,colorlabels

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')
    return img

#comment
def Colorize_func_specific(col,img):
    # col: an int index for which base color is being used
    rgb = colorvals[col]  # grab the rgb for this base color
    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')

    return img

# model training data set and dimensions
data_set_flag = 'padded_mnist_3rd' # mnist, cifar10, padded_mnist, padded_cifar10
imgsize = 28
retina_size = 64 # by default should be same size as image
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = retina_size * retina_size * 3
h_dim1 = 256
h_dim2 = 128
z_dim = 8
l_dim = 64*2 # 2dim (2, retina_size) position
zl_dim = z_dim
sc_dim = 10


#CNN VAE
#this model takes in a single cropped image and a location 1-hot vector  (to be replaced by an attentional filter that determines location from a retinal image)
#there are three latent spaces:location, shape and color and 6 loss functions
#loss functions are: shape, color, location, retinal, cropped (shape + color combined), skip

class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim):
        super(VAE_CNN, self).__init__()
        # encoder part
        self.l_dim = l_dim
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4)*16, h_dim2) #
        self.fc_bn2 = nn.BatchNorm1d(h_dim2) # remove
        # bottle neck part  # Latent vectors mu and sigma
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        self.fc35 = nn.Linear(l_dim, zl_dim)  # location
        self.fc36 = nn.Linear(l_dim, zl_dim)
        self.fc37 = nn.Linear(sc_dim, z_dim)  # scale
        self.fc38 = nn.Linear(sc_dim, z_dim)
        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc4l = nn.Linear(zl_dim, l_dim)  # location
        self.fc4sc = nn.Linear(z_dim, sc_dim)  # scale

        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)
        #self.fc8 = nn.Linear(32*14*14,32*14*14)#16*28*28,16*28*28) #skip conection to hidden dim
        #self.fc9 = nn.Linear(32*14*14,32*14*14)
        self.fc8 = nn.Linear(16*28*28,16*28*28)# #skip conection to hidden dim
        #self.fc9 = nn.Linear(16*28*28,16*28*28)

        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)

        self.skip_bn = nn.BatchNorm2d(16)

        # combine recon and location into retina now using fcs 2dconv and recurrence
        self.fc6 = nn.Linear((imgsize*imgsize*3)+zl_dim, 4000)
        self.fc65 = nn.Linear(4000,4000)#recurrence layer
        self.fc7 = nn.Linear(4000, (retina_size**2)*3)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.1)
        #self.layernorm = nn.LayerNorm(32*14*14)
        self.sparse_relu = nn.Threshold(threshold=0.5, value=0)
        #self.skipconv = nn.Conv2d(16,16,kernel_size=1,stride=1,padding =0,bias=False)

        # map scalars
        self.shape_scale = 1 #1.9
        self.color_scale = 1 #2

    def encoder(self, x, l):
        b_dim = x.size(0)
        l = l.view(b_dim, l_dim)
        h = self.sparse_relu(self.bn1(self.conv1(x)))
        hskip = h.view(b_dim,-1)
        h = self.relu(self.bn2(self.conv2(h)))        
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = h.view(-1,int(imgsize / 4) * int(imgsize / 4)*16)
        h = self.relu(self.fc_bn2(self.fc2(h)))

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), self.fc35(l), self.fc36(l), 0, 0, hskip # mu, log_var

    def activations(self, x):
        if type(x) == list or type(x) == tuple:    #passing in a cropped+ location as input
            l = x[2].cuda()
            #sc = x[3].cuda()
            x = x[1].cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)
        else:  #passing in just cropped image
            x = x.cuda()
            #sc = torch.zeros(x.size()[0], sc_dim).cuda()
            l = torch.zeros(x.size()[0], self.l_dim).cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)
        
        z_shape = self.sampling(mu_shape, log_var_shape)
        z_color = self.sampling(mu_color, log_var_color)
        z_location = self.sampling(mu_location, log_var_location)

        return z_shape, z_color, z_location

    def location_encoder(self, l):
        return self.sampling_location(self.fc35(l), self.fc36(l))

    def sampling_location(self, mu, log_var):
        std = (0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #if self.training:
        #    eps = eps * 5
        return mu + eps * std

    def decoder_location(self, z_shape, z_color, z_location):
        h = self.fc4l(z_location)
        return torch.sigmoid(h).view(-1,2,retina_size)

    def decoder_scale(self, z_shape, z_color, z_scale):
        h = self.fc4sc(z_scale)
        return torch.sigmoid(h).view(-1,10)

    def decoder_retinal(self, z_shape, z_color, z_location, z_scale, hskip = None, whichdecode = None):
        # digit recon
        b_dim = z_shape.size(0)
        if whichdecode == 'shape':
            h = (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        elif whichdecode == 'color':
            h = (F.relu(self.fc4c(z_color)) * self.color_scale)
        else:
            h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).detach().view(-1, 3, imgsize, imgsize) #detach conv
        h = torch.sigmoid(h)
        crop_out = h.clone()
        # location vector recon
        l = z_location.detach() #cont. repr of location
        #l = l.view(-1,1,1,self.z_dim)
        l = torch.sigmoid(l)
        #l = l.expand(-1, 3, imgsize, self.z_dim) # reshape to concat
        # shape vector
        #sc = z_scale.detach() #cont. repr of scale
        #sc = sc.view(-1,1,1,self.z_dim)
        #sc = torch.sigmoid(sc)
        #sc = sc.expand(-1, 3, imgsize, self.z_dim) # reshape to concat
        # combine into retina
        h = h.view(b_dim,-1)
        h = torch.cat([h,l], dim = 1)
        h = self.relu(self.fc6(h))
        #print(h.size())
        h = self.relu(self.fc65(h))

        h = self.relu(self.fc65(h))
        
        h = self.fc7(h)
        #print(h.size())
        #print(h.size())
        h = h.view(-1, 3, retina_size, retina_size)

        if self.training:
            return {'recon':torch.sigmoid(h), 'crop':crop_out}
        else:
            return torch.sigmoid(h)

    def decoder_color(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) * self.color_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn6(self.conv6(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn7(self.conv7(h)))
        if self.training:
            h = self.dropout(h)
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_shape(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4s(z_shape)) * self.shape_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn6(self.conv6(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn7(self.conv7(h)))
        if self.training:
            h = self.dropout(h)
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_cropped(self, z_shape, z_color, z_location, hskip=0):
        #print('crop',z_shape.size())
        h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn6(self.conv6(h)))
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.bn7(self.conv7(h)))
        if self.training:
            h = self.dropout(h)
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_skip_cropped(self, z_shape, z_color, z_location, hskip):
        #mu_skip = self.fc8(hskip)
        #log_var_skip = self.fc9(hskip)
        #hskip = self.sampling(mu_skip, log_var_skip)
        h= self.fc8(hskip)#hskip#
        if self.training:
            h = self.dropout(h)
        h = self.relu(self.skip_bn(h.view(-1,16,28,28))) #self.skip_bn(h.view(-1,16,28,28)) self.skip_bn(h.view(-1,32,14,14))
        #h = self.relu(self.fc9(h))
        #h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        #h = self.relu(self.bn5(self.conv5(h.view(-1,16,7,7))))
        #h = self.relu(self.bn6(self.conv6(h.view(-1,64,7,7))))
        #h = self.relu(self.bn7(self.conv7(h.view(-1,32,14,14)))) #
        #ind = h[:,784:].long()
        #h = h[:,:784]
        #h = self.unpool(h.view(-1,28,28),ind.view(-1,28,28))
        h = self.conv8(h.view(-1,16,28,28)).view(-1, 3, imgsize, imgsize) #skip.view(-1,16,28,28)
        return torch.sigmoid(h)

        
    def decoder_skip_retinal(self, z_shape, z_color, z_location, hskip):
        # digit recon
        h = F.relu(hskip)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize).detach()
        h = torch.sigmoid(h)
        # location vector recon
        l = z_location.detach() #cont. repr of location
        l = l.view(-1,1,1,8)
        l = torch.sigmoid(l)
        l = l.expand(-1, 3, imgsize, 8) # reshape to concat
        # combine into retina
        h = torch.cat([h,l], dim = 3)
        b_dim = h.size()[0]*h.size()[2]
        h = h.view(b_dim,-1)
        h = self.relu(self.fc6(h))
        h = self.fc7(h).view(-1,3,imgsize,retina_size)
        return torch.sigmoid(h)

    '''def activations(self, z_shape, z_color, z_location):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape)) + F.relu(self.fc4l(z_location))
        fc4c = self.fc4c(z_color)
        fc4s = self.fc4s(z_shape)
        fc4l = self.fc4l(z_location)
        fc5 = self.fc5(h)
        return fc4c, fc4s, fc4l, fc5
    '''
    def forward_layers(self, l1, l2, layernum, whichdecode):
        hskip = l1
        if layernum == 1:
            h = F.relu(self.bn2(self.conv2(l1)))
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 2:
            h = self.relu(self.bn3(self.conv3(l2)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 3:
            h=hskip
            #hskip = F.relu(hskip)
            #h = self.relu(self.fc9(hskip))
            #l1 = F.relu(self.fc9(hskip))
            #ind = hskip[:,784:].long()
            #h = hskip[:,:784]
            #l1 = self.unpool(h.view(-1,28,28),ind.view(-1,28,28))
            h = self.relu(self.bn2(self.conv2(h.view(-1,16,28,28)))) ####
            
            h = h.view(-1,32,14,14)
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            mu_shape = self.fc31(h)
            #print(f'{whichdecode}',l1.size()[0],mu_shape.size())
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape, z_color, 0, hskip)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(z_shape, z_color, 0, hskip)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, whichdecode='noskip', keepgrad=[]):
        if type(x) == list or type(x) == tuple:    #passing in a cropped+ location as input
            l = x[2].cuda()
            #sc = x[3].cuda()
            x = x[1].cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)
        else:  #passing in just cropped image
            x = x.cuda()
            #sc = torch.zeros(x.size()[0], sc_dim).cuda()
            l = torch.zeros(x.size()[0], self.l_dim).cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)

        #what maps are used in the training process.. the others are detached to zero out those gradients
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()

        if ('location' in keepgrad):
            z_location = self.sampling_location(mu_location, log_var_location)
        else:
            z_location = self.sampling_location(mu_location, log_var_location).detach()

        if ('skip' in keepgrad):
            hskip = hskip
        else:
            hskip = hskip.detach()

        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape,z_color, z_location, hskip)
        elif (whichdecode == 'retinal'):
            output = self.decoder_retinal(z_shape,z_color, z_location, z_scale=0)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(0, 0, 0, hskip)
        elif (whichdecode == 'skip_retinal'):
            output = self.decoder_skip_retinal(0, 0, z_location, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color , 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape,0, 0)
        elif (whichdecode == 'location'):
            output = self.decoder_location(0, 0, z_location)
        elif (whichdecode == 'scale'):
            output = self.decoder_scale(0, 0, 0, z_scale=0)
        return output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location, mu_scale, log_var_scale

# function to build an  actual model instance
# function to build a model instance
def vae_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1, h_dim2 = h_dim2, z_dim = z_dim, l_dim = l_dim, sc_dim = sc_dim):
    vae = VAE_CNN(x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim)

    folder_path = f'sample_{vae_type}_{data_set_flag}'

    return vae, z_dim

########Actually build it
#vae, z_dim = vae_builder()

#######what optimier to use:
# learning rate = 0.0001
#optimizer = torch.optim.SGD(vae.parameters(), lr=0.0001, momentum = 0.9)
#optimizer = optim.Adam(vae.parameters(), lr=0.0001)
#device = torch.device('cuda:3')


######the loss functions
#Pixelwise loss for the entire retina (dimensions are cropped image height x retina_size)
def loss_function(recon_x, x, crop_x, mu, log_var, mu_c, log_var_c):
    if crop_x is not None:
        x = place_crop(crop_x,x[2].clone())
    else:
        x=x[0].clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3, retina_size, retina_size), x.view(-1, 3, retina_size, retina_size), reduction='sum')
    return BCE

#pixelwise loss for just the cropped image
def loss_function_crop(recon_x, x, mu, log_var, mu_c, log_var_c):
    if len(x) <= 5:
        x = x[1].clone().cuda()
    else:
        x = x.clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), x.view(-1, imgsize * imgsize * 3), reduction='sum')
    return BCE


# loss for shape in a cropped image
def loss_function_shape(recon_x, x, mu, log_var):
    if len(x) <= 5:
        x = x[1].clone().cuda()
    else:
        x = x.clone().cuda()
    # make grayscale reconstruction
    gray_x = x.view(-1, 3, imgsize, imgsize).mean(1)
    gray_x = torch.stack([gray_x, gray_x, gray_x], dim=1)
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), gray_x.view(-1,imgsize * imgsize * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD

#loss for just color in a cropped image
def loss_function_color(recon_x, x, mu, log_var):
    if len(x) <= 5:
        x = x[1].clone().cuda()
    else:
        x = x.clone().cuda()
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(-1, 3 * imgsize * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, imgsize * imgsize * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just location
def loss_function_location(recon_x, x, mu, log_var):
    x = x[2].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1,2,retina_size), x.view(-1,2,retina_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just scale
def loss_function_scale(recon_x, x, mu, log_var):
    x = x[3].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1,retina_size,retina_size), x.view(-1,retina_size,retina_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# test recreate img with different features
def progress_out(vae, data, epoch, count, skip = False, filename = None):
    sample_size = 25
    vae.eval()
    #make a filename if none is provided
    if filename == None:
        filename = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png'
        filename1 = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_crop_{str(count).zfill(5)}.png'

    if skip:
        sample = data[:sample_size]
        with torch.no_grad():
            shape_color_dim = imgsize
            reconds, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'skip_cropped') #digit from skip
            recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
            reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
            recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape
            utils.save_image(
            torch.cat([sample.view(sample_size, 3, imgsize, shape_color_dim).cuda(), reconds.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recond.view(sample_size, 3, imgsize, shape_color_dim).cuda(),
                    reconc.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recons.view(sample_size, 3, imgsize, shape_color_dim).cuda()], 0),
            filename,
            nrow=sample_size, normalize=False, range=(-1, 1),)

    else:
        sample = data
        #print('\n',len(sample))
        with torch.no_grad():
            reconl, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'location') #location
            reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'retinal') #retina
            recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'cropped') #digit
            reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'color') #color
            recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'shape') #shape

        crop_retina = place_crop(reconb['crop'].cuda(), sample[2].cuda())
        reconb = reconb['recon'].cuda()
        loc_background = torch.zeros(sample_size,3,retina_size-2,retina_size).cuda()
        line1 = torch.ones((1,2)) * 0.5
        line1 = line1.view(1,1,1,2)
        line2 = line1.view(1,1,1,2)
        #line3 = line1.expand(sample_size, 3, 2, 2).cuda()
        line1 = line1.expand(sample_size, 3, imgsize, 2).cuda()
        line2 = line2.expand(sample_size, 3, retina_size, 2).cuda()
        

        reconl = reconl.view(sample_size,1,2,retina_size)
        reconl = reconl.expand(sample_size,3,2,retina_size)
        n_reconc = torch.cat((reconc,line1),dim = 3).cuda()
        n_recons = torch.cat((recons,line1),dim = 3).cuda()
        n_reconl = torch.cat((reconl,loc_background),dim = 2).cuda()
        n_reconl = torch.cat((n_reconl,line2),dim = 3).cuda()
        n_recond = torch.cat((recond,line1),dim = 3).cuda()
        crop_retina = torch.cat((crop_retina.cuda(),line2.cuda()),dim = 3).cuda()
        shape_color_dim = retina_size + 2
        shape_color_dim1 = imgsize + 2
        sample = torch.cat((sample[0].cuda(),line2),dim = 3).cuda()
        reconb = torch.cat((reconb,line2.cuda()),dim = 3).cuda()

        utils.save_image(
            torch.cat([sample.view(sample_size, 3, retina_size, shape_color_dim)[:25], crop_retina.view(sample_size, 3, retina_size, shape_color_dim)[:25], reconb.view(sample_size, 3, retina_size, shape_color_dim)[:25], n_reconl.view(sample_size, 3, retina_size, shape_color_dim)[:25]], 0),
            filename,
            nrow=sample_size, normalize=False)

        utils.save_image(
            torch.cat([n_recond.view(sample_size, 3, imgsize, shape_color_dim1)[:25], n_reconc.view(sample_size, 3, imgsize, shape_color_dim1)[:25], n_recons.view(sample_size, 3, imgsize, shape_color_dim1)[:25]], 0),
            filename1,
            nrow=sample_size, normalize=False)

def test_loss(vae, test_data, whichdecode = []):
    loss_dict = {}

    for decoder in whichdecode:
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location, mu_scale, log_var_scale = vae(test_data, decoder)
        
        if decoder == 'retinal':
            loss = loss_function(recon_batch['recon'], test_data, None, mu_shape, log_var_shape, mu_color, log_var_color)
        
        elif decoder == 'cropped':
            loss = loss_function_crop(recon_batch, test_data[1], mu_shape, log_var_shape, mu_color, log_var_color)
        
        loss_dict[decoder] = loss.item()

    return loss_dict

def update_seen_labels(batch_labels, current_labels):
    new_label_lst = []
    for i in range(len(batch_labels)):
        s = batch_labels[0][i].item() # shape label
        c = batch_labels[1][i].item() # color label
        r = batch_labels[2][i].item() # retina location label
        new_label_lst += [(s, c, r)]
    seen_labels = set(new_label_lst) | set(current_labels) # creates a new set 
    return seen_labels

def place_crop(crop_data,loc): # retina placement on GPU for training
    #print(loc.size())
    b_dim = crop_data.size(0)
    out_retina = torch.zeros(b_dim,3,retina_size,retina_size).cuda()
    for i in range(len(out_retina)):
        j,x = torch.max(loc[i][0],dim=0)
        z,y = torch.max(loc[i][1],dim=0)
        #print(x,y)
        out_retina[i,:,(retina_size-y)-imgsize:retina_size-y,x:x+imgsize] = crop_data[i]
    #print(out_retina.size())
    return out_retina

def component_to_grad(comp): # determine gradient for componeent training
    if comp == 'shape':
        return ['shape']
    elif comp == 'color':
        return ['color']
    elif comp == 'cropped':
        return ['shape', 'color']
    elif comp == 'skip_cropped':
        return ['skip']
    elif comp == 'retinal':
        return []
    elif comp == 'location':
        return ['location']
    else:
        raise Exception(f'Invalid component: {comp}')

def train(vae, optimizer, epoch, dataloaders, return_loss = False, seen_labels = {}, components = {}):
    vae.train()
    train_loader_noSkip, emnist_skip, fmnist_skip, test_loader, sample_loader, block_loader = dataloaders[0], dataloaders[1], dataloaders[2], dataloaders[3], dataloaders[4], dataloaders[5]
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) # the latent space is trained on EMNIST, MNIST, and f-MNIST
    block_iter = iter(block_loader)
    if fmnist_skip != None:
        #dataiter_emnist_skip= iter(emnist_skip) # The skip connection is trained on pairs from EMNIST, MNIST, and f-MNIST composed on top of each other
        dataiter_fmnist_skip= iter(fmnist_skip)
    test_iter = iter(test_loader)
    #sample_iter = iter(sample_loader)
    count = 0
    max_iter = 600
    loader=tqdm(train_loader_noSkip, total = max_iter)

    retinal_loss_train, cropped_loss_train = 0, 0 # loss metrics returned to Training.py
    block_data, block_labels = next(block_iter)

    for i,j in enumerate(loader):
        count += 1
        data, batch_labels = next(dataiter_noSkip)

        # shuffle in the block dataset
        z = random.randint(0,10)
        if z <= 1:
            data = block_data
        
        optimizer.zero_grad()
        
        # determine which component is being trained
        comp_ind = count % len(components)
        whichdecode_use = components[comp_ind]
        keepgrad = component_to_grad(whichdecode_use)

        if whichdecode_use == 'skip_cropped':
            data_skip = next(dataiter_fmnist_skip)
            r = random.randint(0,1)
            if r == 1:
                data = data_skip[0]
            else:
                data = data[1]
        
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location, mu_scale, log_var_scale = vae(data, whichdecode_use, keepgrad)
            
        if whichdecode_use == 'shape':  # shape
            loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)

        elif whichdecode_use == 'color': # color
            loss = loss_function_color(recon_batch, data, mu_color, log_var_color)

        elif whichdecode_use == 'location': # location
            loss = loss_function_location(recon_batch, data, mu_location, log_var_location)

        elif whichdecode_use == 'retinal': # retinal
            loss = loss_function(recon_batch['recon'], data, recon_batch['crop'], mu_shape, log_var_shape, mu_color, log_var_color)
            retinal_loss_train = loss.item()

        elif whichdecode_use == 'cropped': # cropped
            loss = loss_function_crop(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
            cropped_loss_train = loss.item()

        elif whichdecode_use == 'skip_cropped': # skip training
            loss = loss_function_crop(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)

        elif whichdecode_use == 'scale': # scale training
            loss = loss_function_crop(recon_batch, data, mu_scale, log_var_scale)
        
        #l1_norm = sum(p.abs().sum() for p in vae.parameters())
        #loss += l1_norm*0.0001
        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description((f'epoch: {epoch}; mse: {loss.item():.5f};'))
        seen_labels = None #update_seen_labels(batch_labels,seen_labels)
        #if count % (0.8*max_iter) == 0:
          #  data, labels = next(sample_iter)
           # progress_out(vae, data, epoch, count)
        #elif count % 500 == 0: not for RED GREEN
         #   data = data_noSkip[0][1] + data_skip[0]
          #  progress_out(vae, data, epoch, count, skip= True)
        
        if i == max_iter +1:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))
    
    if return_loss is True:
        # get test losses for cropped and retinal
        test_data = next(test_iter)
        test_data = test_data[0]

        test_loss_dict = test_loss(vae, test_data, ['retinal', 'cropped'])
    
        return [retinal_loss_train, test_loss_dict['retinal'], cropped_loss_train, test_loss_dict['cropped']], seen_labels

#compute avg loss of retinal recon w/ skip, w/o skip, increase fc?
def test(whichdecode, test_loader_noSkip, test_loader_skip, bs):
    vae.eval()
    global numcolors
    test_loss = 0
    testiter_noSkip = iter(test_loader_noSkip)  # the latent space is trained on MNIST and f-MNIST
    testiter_skip = iter(test_loader_skip)  # The skip connection is trained on notMNIST
    with torch.no_grad():
        for i in range(1, len(test_loader_noSkip)): # get the next batch


            data = testiter_noSkip.next()
            data = data[0]
            recon, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(data, 'retinal')

            # sum up batch loss
            #test_loss += loss_function_shape(recon, data, mu_shape, log_var_shape).item()
            #test_loss += loss_function_color(recon, data, mu_color, log_var_color).item()
            test_loss += loss_function(recon, data, mu_shape, log_var_shape, mu_color, log_var_color).item()

    print('Example reconstruction')
    datac = data[0].cuda()
    datac=datac.view(bs, 3, imgsize, retina_size)
    save_image(datac[0:8], f'{args.dir}/orig.png')
    pos = torch.zeros((64,100)).cuda()
    for i in range(len(pos)):
        pos[i][random.randint(0,99)] = 1
    pos_mu = vae.fc35(pos)
    pos_logvar = vae.fc36(pos)

    # current imagining of shape and color results in random noise
    # generate a
    print('Imagining a shape')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 0
        zs = torch.randn(64, z_dim).cuda() * 1
        zl = vae.sampling(pos_mu, pos_logvar)
        sample = vae.decoder_retinal(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/sampleshape.png')


    print('Imagining a color')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 1
        zs = torch.randn(64, z_dim).cuda() * 0
        zl = vae.sampling(pos_mu, pos_logvar)
        sample = vae.decoder_retinal(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/samplecolor.png')

    test_loss /= len(test_loader_noSkip.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

'''def activations(image, l= None):
    if l is None:
        l = torch.zeros(image.size()[0], vae.l_dim).cuda()
    mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location,j,j, hskip = vae.encoder(image, l)
    l1_act = hskip
    l2_act = hskip
    shape_act = vae.sampling(mu_shape, log_var_shape)
    color_act = vae.sampling(mu_color, log_var_color)
    location_act = vae.sampling_location(mu_location, log_var_location)
    return l1_act , l2_act, shape_act, color_act, location_act#.view(-1,16,28,28)

def image_activations(image, l = None):
    if l is None:
        l = torch.zeros(image.size()[0], vae.l_dim).cuda()
    mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, a,b, hskip = vae.encoder(image, l)
    shape_act = vae.sampling(mu_shape, log_var_shape)
    color_act = vae.sampling(mu_color, log_var_color)
    location_act = vae.sampling_location(mu_location, log_var_location)
    return shape_act, color_act, location_act

def activation_fromBP(L1_activationBP, L2_activationBP, layernum):
    if layernum == 1:
        l2_act_bp = F.relu(vae.fc2(L1_activationBP))
        mu_shape = (vae.fc31(l2_act_bp))
        log_var_shape = (vae.fc32(l2_act_bp))
        mu_color = (vae.fc33(l2_act_bp))
        log_var_color = (vae.fc34(l2_act_bp))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    elif layernum == 2:
        mu_shape = (vae.fc31(L2_activationBP))
        log_var_shape = (vae.fc32(L2_activationBP))
        mu_color = (vae.fc33(L2_activationBP))
        log_var_color = (vae.fc34(L2_activationBP))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    return shape_act_bp, color_act_bp'''


