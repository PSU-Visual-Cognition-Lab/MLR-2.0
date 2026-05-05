colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]
object_names = ['airplane', 'bird', 'car', 'cat', 'dog', 'duck', 'frog', 'horse', 'sailboat', 'truck']
DATASET_ROOT = '/home/bwyble/data/'

# prerequisites
import torch
import sys
import os
from collections import defaultdict
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
from scipy import stats
import gc
from PIL import Image
from itertools import cycle
from MLR_src.dataset_builder import Colorize_specific
import numpy as npy
import seaborn as sns
import joblib
import inspect

#internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MLR_src.dataset_builder import Dataset, Colorize_specific
from MLR_src.mVAE import VAE_CNN
from MLR_src.BP_functions import BPTokens_binding_all, BPTokens_retrieveByToken, BPTokens_storage, BPTokens_with_labels
import random


from PIL import Image, ImageOps, ImageEnhance#, __version__ as PILLOW_VERSION
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA')
else:
    device = 'cpu'

bs_testing = 1000     # number of images for testing. 20000 is the limit
shape_coeff = 1       #cofficient of the shape map
color_coeff = 1       #coefficient of the color map
location_coeff = 0    #Coefficient of Location map
l1_coeff = 1          #coefficient of layer 1
l2_coeff = 1          #coefficient of layer 2
shapeLabel_coeff= 1   #coefficient of the shape label
colorLabel_coeff = 1  #coefficient of the color label
location_coeff = 0  #coefficient of the color label

bpsize = 10000#00         #size of the binding pool
token_overlap =0.1
bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

normalize_fact_familiar=1
normalize_fact_novel=1


imgsize = 28
BP_std = 0

# helper functions:
def log_function_name():
    return inspect.stack()[1].function

def location_to_onehot(locations):
    pass

def onehot_to_location(onehots):
    pass

def compute_correlation(x, y):
    assert x.shape == y.shape, "Tensors must have the same shape"
    
    # Flatten tensors if they're multidimensional
    x = x.view(-1)
    y = y.view(-1)
    #x = replace_near_zero(x)
    #y = replace_near_zero(y)
    
    # Compute means
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    # Compute the numerator
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    
    # Compute the denominator
    x_var = torch.sum((x - x_mean)**2)
    y_var = torch.sum((y - y_mean)**2)
    denominator = torch.sqrt(x_var * y_var)
    
    # Compute correlation
    correlation = numerator / denominator
    
    return correlation


# figures:
@torch.no_grad()
def fig_efficient_rep(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()

    if load_data is False:
        print('generating Figure efficient reconstruction plot')
        bpsize = 2500
        token_overlap = 0.15
        bpPortion = int(token_overlap * bpsize)
        numimg = 7
        n_2 = 1
        n_4 = 4

        # familiar = emnist A-D (trained on), novel = mnist 1-4 (not trained on)
        test_loader_familiar = cycle(Dataset('emnist',{'colorize':True, 'target_set':[0,1,2,3]}, train=True).get_loader(numimg))
        test_loader_novel = cycle(Dataset('mnist',{'colorize':True, 'target_set':[1,2,3,4]}, train=True).get_loader(numimg))

        dataiter_familiar = iter(test_loader_familiar)
        dataiter_novel = iter(test_loader_novel)
        sc_2 = []
        l1_2 = []
        sc_4 = []
        l1_4 = []

        for count in range(0, 100):
            data_familiar, labels = next(dataiter_familiar)
            data_novel, labels = next(dataiter_novel)

            familiar_sample = data_familiar[:n_4].cuda()
            novel_sample = data_novel[:n_4].cuda()

            familiar_act = vae.activations(familiar_sample.view(-1, 3, 28, 28), False)
            novel_act = vae.activations(novel_sample.view(-1, 3, 28, 28), False)

            familiar_shape_act = familiar_act['shape']
            familiar_color_act = familiar_act['color']
            novel_l1_act = novel_act['skip']

            BP_activations_sc_2 = {'shape': [familiar_shape_act[:n_2].view(n_2, -1), 1], 'color': [familiar_color_act[:n_2].view(n_2, -1), 1]}
            BP_activations_l1_2 = {'l1': [novel_l1_act[:n_2].view(n_2, -1), 1]}
            BP_activations_sc_4 = {'shape': [familiar_shape_act.view(n_4, -1), 1], 'color': [familiar_color_act.view(n_4, -1), 1]}
            BP_activations_l1_4 = {'l1': [novel_l1_act.view(n_4, -1), 1]}

            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc_2, n_2, normalize_fact_novel)
            BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc_2, n_2, normalize_fact_novel)
            shape_out_2, color_out_2 = BP_activations_out['shape'], BP_activations_out['color']

            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1_2, n_2, normalize_fact_novel)
            BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1_2, n_2, normalize_fact_novel)
            l1_out_2 = BP_activations_out['l1']

            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc_4, n_4, normalize_fact_novel)
            BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc_4, n_4, normalize_fact_novel)
            shape_out_4, color_out_4 = BP_activations_out['shape'], BP_activations_out['color']

            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1_4, n_4, normalize_fact_novel)
            BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1_4, n_4, normalize_fact_novel)
            l1_out_4 = BP_activations_out['l1']

            recon_sc_2 = vae.decoder_cropped(shape_out_2, color_out_2, 0, 0).cuda()
            recon_l1_2 = vae.decoder_skip_cropped(0, 0, 0, l1_out_2).cuda()
            recon_sc_4 = vae.decoder_cropped(shape_out_4, color_out_4, 0, 0).cuda()
            recon_l1_4 = vae.decoder_skip_cropped(0, 0, 0, l1_out_4).cuda()

            corr_sc_2 = compute_correlation(familiar_sample[:n_2], recon_sc_2).item()
            corr_l1_2 = compute_correlation(novel_sample[:n_2], recon_l1_2).item()
            corr_sc_4 = compute_correlation(familiar_sample, recon_sc_4).item()
            corr_l1_4 = compute_correlation(novel_sample, recon_l1_4).item()

            sc_2 += [corr_sc_2]
            l1_2 += [corr_l1_2]
            sc_4 += [corr_sc_4]
            l1_4 += [corr_l1_4]

        corr_sc_2 = sum(sc_2) / len(sc_2)
        corr_l1_2 = sum(l1_2) / len(l1_2)
        corr_sc_4 = sum(sc_4) / len(sc_4)
        corr_l1_4 = sum(l1_4) / len(l1_4)

        print(corr_l1_2, corr_l1_4)
        print(corr_sc_2, corr_sc_4)

        e = torch.zeros((1, 3, 28, 28)).cuda()
        fig_data = [familiar_sample, torch.cat([recon_sc_2, e, e, e], 0), recon_sc_4, novel_sample,
                    torch.cat([recon_l1_2, e, e, e], 0), recon_l1_4]

        data_to_pickle = {
            "fig_data": fig_data,
            "n_2": n_2,
            "n_4": n_4,
            "corr_l1_2": corr_l1_2,
            "corr_l1_4": corr_l1_4,
            "corr_sc_2": corr_sc_2,
            "corr_sc_4": corr_sc_4,
        }
        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        loaded_data = joblib.load(pkl_path)
        n_2 = loaded_data["n_2"]
        n_4 = loaded_data["n_4"]
        corr_l1_2 = loaded_data["corr_l1_2"]
        corr_l1_4 = loaded_data["corr_l1_4"]
        corr_sc_2 = loaded_data["corr_sc_2"]
        corr_sc_4 = loaded_data["corr_sc_4"]

    save_image(
        torch.cat(fig_data, 0),
        f'{folder_path}efficient_recon_sample.png', pad_value=0.6,
        nrow=n_4, normalize=False)

    plt.figure()
    sns.lineplot(x=[n_2, n_4], y=[corr_l1_2, corr_l1_4], label='novel images (L1)')
    sns.lineplot(x=[n_2, n_4], y=[corr_sc_2, corr_sc_4], label='familiar images (feature maps)')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.title('Set Size vs. Reconstruction Correlation')
    plt.legend()
    plt.savefig(f'{folder_path}efficient_recon.png')
    plt.close()

@torch.no_grad()
def fig_repeat_recon(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    print(bpsize)
    vae.eval()
    print('generating Figure repeated reconstructions, red A, red A, blue C using map and L1')
    retina_size = 100
    imgsize = 28
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    numimg = 3  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained

    test_loader_noSkip= Dataset('emnist',{'colorize':False, 'target_set':[0, 1, 2, 3]}, train=True).get_loader(numimg)

    dataiter_noSkip = iter(test_loader_noSkip)
    data, labels = next(dataiter_noSkip)
    data = data #.cuda()



    # find 2 B's and then 1 C
    imgs = []
    c = 0
    while c < 2:
        if labels[0][0].item() == 0:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    c = 0
    while c < 1:
        if labels[0][0].item() == 2:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    red = Colorize_specific(0)
    blue = Colorize_specific(1)

    imgs[0] = convert_tensor(red(convert_image(imgs[0])))
    imgs[1] = convert_tensor(blue(convert_image(imgs[1])))
    imgs[2] = convert_tensor(blue(convert_image(imgs[2])))
    
    sample = torch.cat(imgs, 0).cuda()
    
    #push the images through the model
    activations = vae.activations(sample.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']
    reconb = vae.decoder_cropped(shape_act, color_act, 0)
    reconskip = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(numimg,-1))
    #reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample.view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample.view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act

    BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}
    # store 1 -> numimg items
    for n in range(numimg,numimg+1):
        #Store and retrieve the map versions
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n,normalize_fact_novel)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        z = torch.randn(numimg-n,8).cuda()
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()
        #retrievals = retrievals[:n]
        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n,normalize_fact_novel)
        #l1_out_all=l1_act[:n] #remove
        l1_out_all = BP_activations_out['l1']
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n,-1))

        #imgmatrixMap= torch.cat([imgmatrixMap] + retrievals,0)
        
        imgmatrixMap= torch.cat([imgmatrixMap, retrievals],0)
        imgmatrixL1= torch.cat([imgmatrixL1,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrixMap= torch.cat([imgmatrixMap,emptyshape*0],0)
            imgmatrixL1= torch.cat([imgmatrixL1,emptyshape*0],0)
    #each figure is the raw image, then a reconstruction from the BP then direct reconstruction, then memory
    save_image(imgmatrixL1, f'{folder_path}figure_repeat_L1.png',  nrow=numimg,        normalize=False) #range=(-1, 1))
    save_image(imgmatrixMap, f'{folder_path}figure_repeat_Map.png',  nrow=numimg,        normalize=False) #,range=(-1, 1))

@torch.no_grad()
def fig_non_repeat_recon(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    #this is the individuated simulation
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    vae.eval()
    print('generating Figure repeated reconstructions, green any, red any, red any')
    retina_size = 100
    imgsize = 28
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    numimg = 3  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    #test_loader_noSkip= Dataset('emnist',{'colorize':False}, train=True).get_loader(numimg)  
    test_loader_noSkip= Dataset('emnist',{'colorize':False, 'target_set':[0, 1, 2, 3]}, train=True).get_loader(numimg)
    dataiter_noSkip = iter(test_loader_noSkip)
    data, labels = next(dataiter_noSkip)
    data = data #.cuda()

    #grab 3 random items
    imgs = []
    c = 0
    while c < 2:
        imgs += [data[0]]
        c += 1
        data, labels = next(dataiter_noSkip)
    
    c = 0
    while c < 1:
        imgs += [data[0]]
        c += 1
        data, labels = next(dataiter_noSkip)
    
    blue = Colorize_specific(0)
    green = Colorize_specific(1)

    imgs[0] = convert_tensor(blue(convert_image(imgs[0])))
    imgs[1] = convert_tensor(green(convert_image(imgs[1])))
    imgs[2] = convert_tensor(green(convert_image(imgs[2])))
    
    sample = torch.cat(imgs, 0).cuda()
    
    #push the images through the model
    activations = vae.activations(sample.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']
    reconb = vae.decoder_cropped(shape_act, color_act, 0)
    reconskip = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(numimg,-1))
    #reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample.view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample.view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act

    BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}
    # store 1 -> numimg items
    for n in range(numimg,numimg+1):
        #Store and retrieve the map versions
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n,normalize_fact_novel)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        z = torch.randn(numimg-n,8).cuda()
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()
        #retrievals = retrievals[:n]
        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n,normalize_fact_novel)
        #l1_out_all=l1_act[:n] #remove
        l1_out_all = BP_activations_out['l1']
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n,-1))

        #imgmatrixMap= torch.cat([imgmatrixMap] + retrievals,0)
        
        imgmatrixMap= torch.cat([imgmatrixMap, retrievals],0)
        imgmatrixL1= torch.cat([imgmatrixL1,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrixMap= torch.cat([imgmatrixMap,emptyshape*0],0)
            imgmatrixL1= torch.cat([imgmatrixL1,emptyshape*0],0)
    
    save_image(imgmatrixL1, f'{folder_path}figure_non_repeat_L1.png',  nrow=numimg,        normalize=False) #range=(-1, 1))
    save_image(imgmatrixMap, f'{folder_path}figure_non_repeat_Map.png',  nrow=numimg,        normalize=False) #,range=(-1, 1))


@torch.no_grad()
def fig_non_color_repeat_recon(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    vae.eval()
    print('generating Figure repeated reconstructions, any 5, any 5, any 3')
    retina_size = 100
    imgsize = 28
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    numimg = 3  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    test_loader_noSkip= Dataset('emnist',{'colorize':False, 'target_set':[0, 1, 2, 3]}, train=True).get_loader(numimg)
    #test_loader_noSkip= Dataset('emnist',{'colorize':False}, train=True).get_loader(numimg)  
    #test_loader_noSkip = cycle(Dataset('emnist',{'colorize':False, 'target_set':[0, 1, 2, 3]}, train=True).get_loader(numimg))

    dataiter_noSkip = iter(test_loader_noSkip)
    data, labels = next(dataiter_noSkip)
    data = data #.cuda()

    # find 2 a's and then 1 b
    imgs = []
    c = 0
    while c < 2:
        if labels[0][0].item() == 1:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    c = 0
    while c < 1:
        if labels[0][0].item() == 2:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    blue = Colorize_specific(random.randint(0,9))
    green = Colorize_specific(random.randint(0,9))
    green1 = Colorize_specific(random.randint(0,9))

    imgs[0] = convert_tensor(blue(convert_image(imgs[0])))
    imgs[1] = convert_tensor(green(convert_image(imgs[1])))
    imgs[2] = convert_tensor(green1(convert_image(imgs[2])))
    
    sample = torch.cat(imgs, 0).cuda()
    
    #push the images through the model
    activations = vae.activations(sample.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']
    reconb = vae.decoder_cropped(shape_act, color_act, 0)

    reconskip = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(numimg,-1))
    #reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample.view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample.view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act

    BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}
    # store 1 -> numimg items
    for n in range(numimg,numimg+1):
        #Store and retrieve the map versions
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n,normalize_fact_novel)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        z = torch.randn(numimg-n,8).cuda()
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()
        #retrievals = retrievals[:n]
        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n,normalize_fact_novel)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n,normalize_fact_novel)
        #l1_out_all=l1_act[:n] #remove
        l1_out_all = BP_activations_out['l1']
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n,-1))

        #imgmatrixMap= torch.cat([imgmatrixMap] + retrievals,0)
        
        imgmatrixMap= torch.cat([imgmatrixMap, retrievals],0)
        imgmatrixL1= torch.cat([imgmatrixL1,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrixMap= torch.cat([imgmatrixMap,emptyshape*0],0)
            imgmatrixL1= torch.cat([imgmatrixL1,emptyshape*0],0)
    
    save_image(imgmatrixL1, f'{folder_path}figure_non_color_repeat_L1.png',  nrow=numimg,        normalize=False) #range=(-1, 1))
    save_image(imgmatrixMap, f'{folder_path}figure_non_color_repeat_Map.png',  nrow=numimg,        normalize=False) #,range=(-1, 1))


@torch.no_grad()
def fig_novel_representations(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    #from dataset_builder import Colorize_specific
    all_imgs = []
    print('generating Figure 2b, Novel characters retrieved from memory of L1 and Bottleneck')
    retina_size = 100
    imgsize = 28
    numimg = 6
    vae.eval()
    bpsize = 10000#00         #size of the binding pool
    token_overlap =0.35
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        color = Colorize_specific(random.randint(0,9))
        img = Image.open(f'{DATASET_ROOT}current_bengali/{i}_thick.png')# Image.open(f'change_image_{i}.png') #
        img = img.resize((28, 28))
        img_new = color(img)   # Currently broken, but would add a color to each
        img_new = convert_tensor(img_new)[0:3,:,:]
        #img_new[0] = torch.zeros_like(img_new[0])
        #img_new[2] = torch.zeros_like(img_new[1])

        all_imgs.append(img_new)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()
    #location = torch.zeros(imgs.size()[0], vae.l_dim).cuda()
    #location[0] = 1

    blank = torch.zeros(1,3,28,28).cuda()
    blank[:,:,]
    #push the images through the encoder
    activations = vae.activations(imgs.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']

    imgmatrixL1skip  = torch.empty((0,3,28,28)).cuda()
    imgmatrixL1noskip  = torch.empty((0,3,28,28)).cuda()
    imgmatrixMap  = torch.empty((0,3,28,28)).cuda()
    
    
    #now run them through the binding pool!
    #store the items and then retrive them, and do it separately for shape+color maps, then L1, then L2. 
    #first store and retrieve the shape, color and location maps
    for n in range (0,numimg):
            # reconstruct directly from activation
        #recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped')

        BP_activations_l1 = {'l1': [l1_act[n].view(1,-1), 1]}
        BP_activations_sc = {'shape': [shape_act[n].view(1,-1), 1], 'color': [color_act[n].view(1,-1), 1]}
        
        #now store/retrieve from L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, 1,normalize_fact_novel)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, 1,normalize_fact_novel)
        BP_layerI_out = BP_act_out['l1']
        #print(BP_layerI_out.size())

        BP_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, BP_layerI_out.view(1,-1))

        # reconstruct  from BP version of layer 1, run through the bottleneck
        bn_act = vae.activations(0, False, BP_layerI_out.view(1,-1))
        BP_layer1_noskip = vae.decoder_cropped(bn_act['shape'], bn_act['color'])
        
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, 1,normalize_fact_novel)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, 1,normalize_fact_novel)
        shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
        #reconstruct from BP version of the shape and color maps
        retrievals = vae.decoder_cropped(shape_out_BP, color_out_BP,0,0).cuda()

        imgmatrixL1skip = torch.cat([imgmatrixL1skip,BP_layer1_skip])
        imgmatrixL1noskip = torch.cat([imgmatrixL1noskip,BP_layer1_noskip])
        imgmatrixMap= torch.cat([imgmatrixMap,retrievals])
    
    store2 = []
    

    #save an image showing:  original images, reconstructions directly from L1,  from L1 BP, from L1 BP through bottleneck, from maps BP
    save_image(torch.cat([imgs[0: numimg].view(numimg, 3, 28, imgsize), imgmatrixL1skip], 0), f'{folder_path}fig_novel_rep.png',
            nrow=numimg,            normalize=False,)

@torch.no_grad()
def fig_retinal_mod(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    #the compositional figure
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    vae.eval()
    bs = 10
    emnist_transforms = {'retina':True, 'colorize':True, 'scale':True}
    #    emnist_loader= Dataset('emnist', emnist_transforms).get_loader(bs)
    emnist_loader = Dataset('emnist', {'retina':True, 'colorize':True, 'scale':True, 'target_set':[0, 1, 2, 3,15]}, train=True).get_loader(bs)
    
    dataiter_emnist = iter(emnist_loader)
    data, labels = next(dataiter_emnist)
    data = data[0].cuda()

    activations = vae.activations(data, True)
    
    shape = activations['shape']
    color = activations['color']
    scale = activations['scale']
    location = activations['location']

    mod_location = torch.tensor([-0.4, -0.4] * bs).view(bs,2).cuda()
    mod_scale = torch.tensor([0.8] * bs).view(bs,1).cuda()

    theta = torch.cat([scale, location], 1)
    theta_mod_loc = torch.cat([scale, mod_location], 1)
    theta_mod_scale = torch.cat([mod_scale, location], 1)
    theta_mod_both = torch.cat([mod_scale, mod_location], 1)

    recon = vae.decoder_retinal(shape, color, theta)
    recon_mod_loc = vae.decoder_retinal(shape, color, theta_mod_loc)
    recon_mod_scale = vae.decoder_retinal(shape, color, theta_mod_scale)
    recon_mod_both = vae.decoder_retinal(shape, color, theta_mod_both)

    save_image(
        torch.cat([data, recon, recon_mod_scale, recon_mod_loc, recon_mod_both], 0),
        f'{folder_path}figure_retinal_mod.png', pad_value=0.6,
        nrow=bs, normalize=False)

@torch.no_grad()
def fig_visual_synthesis(vae: VAE_CNN, shape_label, s_classes, object_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    #combine a D and a P together to get as sailboat
    vae.eval()
    bs = 2
    num1 = 3
    num2 = 15
    device = next(vae.parameters()).device
    shape_label.to(device)
    num_labels = F.one_hot(torch.tensor([num1, num2]).to(device), num_classes=s_classes).float().to(device) # shape
    print(num_labels.size())
    z_shape = shape_label(num_labels, 1)

    recon_crop = vae.decoder_shape(z_shape)

    location = torch.tensor([[0.0, 0.0], [0.05, -0.3]]).view(bs,2).to(device)
    scale = torch.tensor([[3.1], [0.5]]).view(bs,1).to(device)
    rotation = torch.tensor([[4.0], [0.0]]).view(bs,1).to(device)
    theta = torch.cat([scale, location, rotation], 1)

    recon = vae.decoder_retinal(z_shape, 0, theta, 'shape')

    img1 = recon[0]
    img2 = recon[1]
    comb_img = torch.clamp(img1 + img2, 0, 0.5) * 1.5
    comb_img = comb_img.view(1,3,64,64)

    activations = vae.activations(comb_img, True, None, 'object')

    pred_ss = object_classifier.predict(activations['shape'].cpu())
    out_pred = pred_ss[0]
    pred_prob = object_classifier.predict_proba(activations['shape'].cpu())
    class_idx = list(object_classifier.classes_).index(out_pred)
    out_prob = pred_prob[0][class_idx]

    recon_shape = vae.decoder_object(activations['shape'], 0, 0)
    save_image(comb_img, f'{folder_path}D_P_sim.png')
    save_image(recon_shape, f'{folder_path}D_P_sim_recon.png')
    save_image(recon_crop, f'{folder_path}D_P_crop_recon.png')
    save_image(img1, f'{folder_path}D.png')
    save_image(img2, f'{folder_path}P.png')

    for cls, prob in zip(object_classifier.classes_, pred_prob[0]):
        print(f'{object_names[cls]}: {prob:.4f}')
    print(f'Predicted: {object_names[out_pred]} ({out_prob:.4f})')

def build_gen_grid(joint_recons, shape_recons, color_recons, n):
    grid_rows = []
    empty_block = torch.zeros_like(joint_recons[0])

    for i in range(n + 1):
        row_blocks = []
        for j in range(n + 1):
            if i == 0 and j > 0:
                block = shape_recons[j - 1]
            elif j == 0 and i > 0:
                block = color_recons[i - 1]
            elif i == j:
                block = joint_recons[i]
            else:
                block = empty_block

            if block.dim() == 4 and block.shape[0] == 1:
                block = block.squeeze(0)

            row_blocks.append(block)

        # concat horizontally
        row = torch.cat(row_blocks, dim=2)
        grid_rows.append(row)

    # concat vertically
    return torch.cat(grid_rows, dim=1)
    

@torch.no_grad()
def fig_generative_noise(vae: VAE_CNN, shape_label, s_classes, color_label, c_classes, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()

    if load_data is False:
        print("generative noise plot")
        bs = 2
        x, y= 5, 5
        num1 = 1
        num2 = 6
        device = next(vae.parameters()).device
        shape_label.to(device)
        num_labels = F.one_hot(torch.tensor([num1, num2]).to(device), num_classes=s_classes).float().to(device) # shape
        col_labels = F.one_hot(torch.tensor([1, num2]).to(device), num_classes=c_classes).float().to(device) # color
        print(num_labels.size())
        z_shape_0 = shape_label(num_labels, 1)[0]
        z_color_0 = color_label(col_labels, 1)[0]

        # shape / color noising x: shape, y: color
        recon_crop = vae.decoder_cropped(z_shape_0, z_color_0)
        shape_color_base = recon_crop[0]

        shape_recons = []
        color_recons = []
        joint_recons = [shape_color_base]

        z_shape = z_shape_0.clone()
        z_color = z_color_0.clone()
        n = 9
        for _ in range(0,n):
            z_shape = z_shape + (0.5 * torch.randn_like(z_shape_0))
            z_color = z_color + (1 * torch.randn_like(z_color_0))

            recon_shape_m = vae.decoder_cropped(z_shape, z_color_0)
            recon_color_m = vae.decoder_cropped(z_shape_0, z_color)
            recon_crop_m = vae.decoder_cropped(z_shape, z_color)

            shape_recons += [recon_shape_m]
            color_recons += [recon_color_m]
            joint_recons += [recon_crop_m]

        recon_grid = build_gen_grid(joint_recons, shape_recons, color_recons, n)
        data_to_pickle = {
            "recon_grid": recon_grid,
        }

        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        
        # load plotting data
        loaded_data = joblib.load(pkl_path)
        recon_grid = loaded_data["recon_grid"]
    
    save_image(recon_grid, f'{folder_path}sample.png', pad_value=0.6)

def binding_trial(trial_name: str, dataset, vae: VAE_CNN, color_classifier, numimg, folder_path: str):
    bpsize = 10000
    token_overlap = 0.3
    bpPortion = int(token_overlap *bpsize)

    test_loader = cycle(dataset.get_loader(numimg))
    dataiter = iter(test_loader)
    total_trials = 1000
    out_predictions = 0
    token_predictions = 0
    
    for _ in range(total_trials):
        imgs, targets = next(dataiter)
        imgs = imgs.cuda()

        # greyscale
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=imgs.device).view(1, 3, 1, 1)
        grey = (imgs * weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        grey_imgs = grey.repeat(1, 3, 1, 1)

        #push the images through the encoder
        activations = vae.activations(imgs.view(-1,3,imgsize, imgsize), False)
        shape_act = activations['shape']
        color_act = activations['color']

        grey_activations = vae.activations(grey_imgs.view(-1,3,imgsize, imgsize), False)
        grey_shape_act = grey_activations['shape']
        
        BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
        
        #now store digits
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg, normalize_fact_novel)
        
        BPOut_cued = BPOut.clone()
        # cue by greyscale shape activation of first image
        tokenactivation = torch.zeros(numimg)
        notLink_all = Tokenbindings[0]
        shape_fw = Tokenbindings[1]
        BP_reactivate = torch.mm(grey_shape_act[0].view(1, -1),shape_fw)
        BP_reactivate = BP_reactivate  * BPOut

        for tokens in range(numimg):  # for each token
            BP_reactivate_tok = BP_reactivate.clone()
            BP_reactivate_tok[0,notLink_all[tokens, :]] = 0  # set the BPs to zero for this token retrieval
            # for this demonstration we're assuming that all BP-> token weights are equal to one, so we can just sum the
            # remaining binding pool neurons to get the token activation
            tokenactivation[tokens] = BP_reactivate_tok.sum()

        max, maxtoken =torch.max(tokenactivation,0) #which token has the most activation
        BPOut_cued[0, notLink_all[maxtoken, :]] = 0

        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg, normalize_fact_novel)
        BP_act_out_cued = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut_cued, Tokenbindings, BP_activations_sc, numimg, normalize_fact_novel)
        
        #cued_prediction = shape_classifier.predict(BP_act_out_cued['shape'][0].view(1,-1).cpu())
        #print(targets[0], cued_prediction)
        cued_color_prediction = color_classifier.predict(BP_act_out_cued['color'][maxtoken].view(1,-1).cpu())
        #print(cued_color_prediction, targets[1][0])
        if cued_color_prediction == targets[1][0].item():
            out_predictions += 1
        if maxtoken == 0:
            token_predictions += 1

    print(f"Correct token: {token_predictions/total_trials} of {total_trials} trials")
    print(f"Correct color: {out_predictions/total_trials} of {total_trials} trials")
    with open(f"{folder_path}{trial_name}results.txt", "w") as f:
        f.write(f"Correct token: {token_predictions/total_trials} of {total_trials} trials\n")
        f.write(f"Correct color: {out_predictions/total_trials} of {total_trials} trials\n")

    shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
    shape_out_BP_cued, color_out_BP_cued = BP_act_out_cued['shape'], BP_act_out_cued['color']
    BP_cropped_recon = vae.decoder_cropped(shape_out_BP, color_out_BP)
    BP_cropped_recon_cued = vae.decoder_cropped(shape_out_BP_cued, color_out_BP_cued)
    empty = torch.zeros(1,3,imgsize,imgsize).cuda()
    grey_cue = torch.cat([grey_imgs[0].view(1,3,imgsize,imgsize), empty]).view(numimg, 3, imgsize, imgsize)
    BP_cropped_recon_cued = torch.cat([BP_cropped_recon_cued[0].view(1,3,imgsize,imgsize), empty]).view(numimg, 3, imgsize, imgsize)
    sample = imgs[0: numimg].view(numimg, 3, imgsize, imgsize)

    #save an image showing:  original images, reconstructions directly from L1,  from L1 BP, from L1 BP through bottleneck, from maps BP
    fig_data_list = [sample, BP_cropped_recon, grey_cue, BP_cropped_recon_cued]
    
    # save params used in this simulation run
    with open(f"{folder_path}params.txt", "w") as f:
        f.write(f"bpsize: {bpsize}\n")
        f.write(f"token_overlap: {token_overlap}\n")
        f.write(f"bpPortion: {bpPortion}\n")
    
    return fig_data_list

@torch.no_grad()
def fig_binding_addressability(vae: VAE_CNN, color_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()
    if load_data is False:
        print("addressability figure")
        # store 2 digits, generate activations of greyscaled rep of 1 of the digits, retrieve from BP using that as a cue
        numimg = 2

        # number binding pool neurons used for each item
        dataset = Dataset('emnist',{'retina':False, 'colorize':True, 'rotate':False, 'scale':True}, train=False)
        dataset_2 = Dataset('emnist',{'retina':False, 'colorize':True, 'rotate':False, 'scale':True, 'target_set':[2]}, train=True)
        fig_data_list_r = binding_trial('emnist_rand', dataset, vae, color_classifier, numimg, folder_path)
        fig_data_list_2 = binding_trial('emnist_2', dataset_2, vae, color_classifier, numimg, folder_path)
    
        data_to_pickle = {
            "fig_data_list_r": fig_data_list_r,
            "fig_data_list_c": fig_data_list_2
        }

        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        
        # load plotting data
        loaded_data = joblib.load(pkl_path)
        fig_data_list_r = loaded_data["fig_data_list_r"]
        fig_data_list_2 = loaded_data["fig_data_list_2"]

    save_image(torch.cat(fig_data_list_r, 0), f'{folder_path}mnist_rand-addressability.png',
                nrow=numimg, normalize=False, pad_value=0.6)

    save_image(torch.cat(fig_data_list_2, 0), f'{folder_path}mnist_c-addressability.png',
            nrow=numimg, normalize=False, pad_value=0.6)

def sample_points(n, m, k=5, min_dist=5):
    points = []

    def far_enough(p, q):
        dist = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
        return dist >= min_dist   # Euclidean distance

    attempts = 0
    max_attempts = 10_000

    while len(points) < k and attempts < max_attempts:
        candidate = (random.randint(0, n-1), random.randint(0, m-1))
        if all(far_enough(candidate, p) for p in points):
            points.append(candidate)
        attempts += 1

    if len(points) < k:
        raise RuntimeError("Failed to place points with required spacing.")

    return points

def feature_swap_trial(dataset, vae: VAE_CNN, numimg: int, imgsize: int):
    test_loader = cycle(dataset.get_loader(numimg))
    dataiter = iter(test_loader)
    
    errors_1 = []
    errors_2 = []
    correct_token_err = []
    correct_token_chosen_err = []
    token_swap = 0
    swap_count = 0
    trial_count = 10
    for _ in range(trial_count):
        crop_imgs = next(dataiter)[0].cuda()

        imgs = torch.zeros(numimg,3,64,64).cuda()
        locations = sample_points(64 - imgsize, 64 - imgsize, k=numimg, min_dist=5)
        colors = []
        for i in range(numimg):
            x, y = locations[i]
            imgs[i,:,x:x+imgsize,y:y+imgsize] = crop_imgs[i]
            color = 1 #random.randint(0,9)
            colors.append(color)
            colorizer = Colorize_specific(color)
            frame = convert_tensor(colorizer(convert_image(imgs[i].cpu())))
            imgs[i] = frame.cuda()
        
        excluded_colors = colors.copy()
        crop_imgs_ex = next(dataiter)[0].cuda()
        imgs_ex = torch.zeros(numimg,3,64,64).cuda()
        for i in range(numimg):
            x, y = locations[i]
            imgs_ex[i,:,x:x+imgsize,y:y+imgsize] = crop_imgs_ex[i]
            color = random.randint(0,9)
            colorizer = Colorize_specific(color)
            frame = convert_tensor(colorizer(convert_image(imgs_ex[i].cpu())))
            imgs_ex[i] = frame.cuda()

        #push the images through the encoder
        activations = vae.activations(imgs.view(-1,3,64, 64), True)
        shape_act = activations['shape']
        color_act = activations['color']
        location_act = activations['location']

        cue_activations = vae.activations(imgs[0].view(-1,3,64, 64), True)
        #cue_shape_act = cue_activations['shape']
        cue_location_act = cue_activations['location']

        excluded_activations = vae.activations(imgs_ex.view(-1,3,64, 64), True)
        #excluded_shape_act = excluded_activations['shape']
        excluded_color_act = excluded_activations['color']
        
        BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1],
                             'color': [color_act.view(numimg,-1), 1],
                             'location': [location_act.view(numimg,-1), 1]}
        
        #now store digits
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg, normalize_fact_novel)
        
        BPOut_cued = BPOut.clone()
        # cue by greyscale shape activation of first image
        tokenactivation = torch.zeros(numimg)
        notLink_all = Tokenbindings[0]
        location_fw = Tokenbindings[3]
        BP_reactivate = torch.mm(cue_location_act.view(1, -1),location_fw)
        BP_reactivate = BP_reactivate  * BPOut

        for tokens in range(numimg):  # for each token
            BP_reactivate_tok = BP_reactivate.clone()
            BP_reactivate_tok[0,notLink_all[tokens, :]] = 0
            tokenactivation[tokens] = BP_reactivate_tok.sum()

        max, maxtoken =torch.max(tokenactivation,0) #which token has the most activation
        BPOut_cued[0, notLink_all[maxtoken, :]] = 0

        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg, normalize_fact_novel)
        BP_act_out_cued = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut_cued, Tokenbindings, BP_activations_sc, numimg, normalize_fact_novel)
        
        shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
        shape_out_BP_cued, color_out_BP_cued = BP_act_out_cued['shape'], BP_act_out_cued['color']
        
        print(color_out_BP_cued.size())
        errors = []
        for i in range(numimg):
            errors += [torch.norm(color_act[i]-color_out_BP_cued[0]).item()]
        
        correct_token_err += [errors[0]]
        
        if maxtoken == 0:
            correct_token_chosen_err += [torch.norm(color_act[0]-color_out_BP_cued[0]).item()]
        
        if maxtoken != 0:
            token_swap += 1

        elif errors[0] != min(errors):
            swap_count += 1

        excluded_errors = []
        for i in range(numimg):
            excluded_errors += [torch.norm(excluded_color_act[i]-color_out_BP_cued[0]).item()]
        
        errors_1 += [npy.mean(errors[1:])]
        errors_2 += [npy.mean(excluded_errors)]
    correct_token_err_out = npy.mean(npy.array(correct_token_err))
    correct_token_chosen_err_out = npy.mean(npy.array(correct_token_chosen_err))
    errors = npy.mean(npy.array(errors_1))
    excluded_errors = npy.mean(npy.array(errors_2))
    
    print(locations)
    print(f"Feature swap count: {token_swap} {swap_count} out of {trial_count}")
    print("Feature swap color vector difference:", errors)
    print("Feature swap color vector difference exlcuded:", excluded_errors)
    return [token_swap / trial_count, swap_count / trial_count, correct_token_err_out, correct_token_chosen_err_out, errors, excluded_errors]

@torch.no_grad()
def fig_feature_swap(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    
    if load_data is False:
        vae.eval()
        print("addressability figure")
        # store 2 digits, generate activations of greyscaled rep of 1 of the digits, retrieve from BP using that as a cue

        bpsize = 25000        #size of the binding pool
        token_overlap =0.3
        bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

        dataset = Dataset('square',{'retina':False, 'colorize':False, 'rotate':False, 'scale':True}, train=False)

        #iterate numimg 1-8, compute swap rate at each
        token_swap_rates = []
        color_swap_rates = []
        errors_list = []
        correct_token_chosen_err = []
        for numing in range(1,8):
            swaps = feature_swap_trial(dataset, vae, numing, imgsize=imgsize)
            token_swap_rates += [swaps[0]]
            color_swap_rates += [swaps[1]]
            errors_list += [[swaps[2], swaps[4]]]
            correct_token_chosen_err += [swaps[3]]
        
        data_to_pickle = {
            "token_swap_rates": token_swap_rates,
            "color_swap_rates": color_swap_rates,
            "errors_list": errors_list,
            "correct_token_chosen_err": correct_token_chosen_err
        }

        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        
        # load plotting data
        loaded_data = joblib.load(pkl_path)
        token_swap_rates = loaded_data["token_swap_rates"]
        color_swap_rates = loaded_data["color_swap_rates"]
        errors_list = loaded_data["errors_list"]
        correct_token_chosen_err = loaded_data["correct_token_chosen_err"]

    # % an incorrect token is selected
    sns.lineplot(x=range(1, 8), y=token_swap_rates)
    plt.xlabel('Number of items')
    plt.ylabel('Token swap rate')
    plt.savefig(f'{folder_path}token_swap_rate.png')
    plt.close()
    
    # % the correct token is selected but the incorrect square is chosen
    sns.lineplot(x=range(1, 8), y=color_swap_rates)
    plt.xlabel('Number of items')
    plt.ylabel('Feature swap rate')
    plt.savefig(f'{folder_path}feature_swap_rate.png')
    plt.close()

    # error between selected color and true color  
    sns.lineplot(x=range(1, 8), y=[e[0] for e in errors_list], label='Correct items')
    sns.lineplot(x=range(1, 8), y=[e[1] for e in errors_list], label='Other items')
    sns.lineplot(x=range(1, 8), y=correct_token_chosen_err, label='Correct token chosen')
    plt.xlabel('Number of items')
    plt.ylabel('MSE latent color vector')
    plt.legend()
    plt.savefig(f'{folder_path}feature_error.png')
    plt.close()

@torch.no_grad()
def fig_encoding_flexibility(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    
    vae.eval()
    print("encoding flexibility figure")
    numimg = 2

    bpsize = 25000#00         #size of the binding pool
    token_overlap =0.35
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

    #dataset = Dataset('emnist',{'retina':True, 'colorize':True, 'rotate':False, 'scale':True}, train=False)
    dataset = Dataset('emnist',{'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'target_set':[0, 1, 2, 3]}, train=True)
    test_loader = dataset.get_loader(numimg)
    dataiter = iter(test_loader)
    imgs = next(dataiter)[0][0].cuda()

    #push the images through the encoder
    activations = vae.activations(imgs.view(-1,3,64,64), True)
    shape_act = activations['shape']
    color_act = activations['color']
    location_act = activations['location']
    scale_act = activations['scale']
    
    color_degraded = []
    shape_degraded = []
    
    # degrade shape encoding weight: 1 -> 0.2
    for n in range (1,10,2):
        BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1/n], 'color': [color_act.view(numimg,-1), 1], 
                             'location': [location_act.view(numimg,-1), 1], 'scale': [scale_act.view(numimg,-1), 1]}
        
        #now store/retrieve from L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg,normalize_fact_novel)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg,normalize_fact_novel)
        
        shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
        location_out_BP, scale_out_BP = BP_act_out['location'], BP_act_out['scale']

        theta = torch.cat([scale_out_BP, location_out_BP], 1)
        BP_retinal_recon = vae.decoder_retinal(shape_out_BP, color_out_BP, theta)
        shape_degraded += [BP_retinal_recon]
    
    # degrade color encoding weight: 1 -> 0.2
    for n in range (1,10,2):
        BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1/n], 
                             'location': [location_act.view(numimg,-1), 1], 'scale': [scale_act.view(numimg,-1), 1]}
        
        #now store/retrieve from L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg,normalize_fact_novel)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg,normalize_fact_novel)
        
        shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
        location_out_BP, scale_out_BP = BP_act_out['location'], BP_act_out['scale']

        theta = torch.cat([scale_out_BP, location_out_BP], 1)
        BP_retinal_recon = vae.decoder_retinal(shape_out_BP, color_out_BP, theta)
        color_degraded += [BP_retinal_recon]
    degraded = [x for pair in zip(shape_degraded, color_degraded) for x in pair]
    degraded = torch.cat(degraded,0)
    sample = imgs[0: numimg].view(numimg, 3, 64, 64)

    #save an image showing:  original images, reconstructions directly from L1,  from L1 BP, from L1 BP through bottleneck, from maps BP
    save_image(torch.cat([sample, sample, degraded], 0), f'{folder_path}figure2b.png',
            nrow=numimg*2, normalize=False, pad_value=0.6)
    
    with open(f"{folder_path}params.txt", "w") as f:
        f.write(f"bpsize: {bpsize}\n")
        f.write(f"token_overlap: {token_overlap}\n")
        f.write(f"bpPortion: {bpPortion}\n")