colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]
object_names = ['airplane', 'bird', 'car', 'cat', 'dog', 'duck', 'frog', 'horse', 'sailboat', 'truck', 'clock', 'umbrella']
emnist_labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
DATASET_ROOT = '/home/bwyble/data/'
quickdraw_target_set = [0,2,8,10,11]

# prerequisites
import torch
import sys
import os
from collections import defaultdict
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
from scipy import stats, ndimage
import gc
from PIL import Image
from itertools import cycle
from MLR_src.dataset_builder import Colorize_specific
import numpy as npy
import seaborn as sns
import joblib
import inspect


from skimage.metrics import structural_similarity as ssim

#internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MLR_src.dataset_builder import Dataset, Colorize_specific
from MLR_src.mVAE import VAE_CNN
from MLR_src.BP_functions import BPTokens_binding_all, BPTokens_retrieveByToken, BPTokens_storage, BPTokens_with_labels, BPTokens_storage_bitmask, BPTokens_retrieveByToken_bitmask
from MLR_src.label_network import s_classes, VAEshapelabels
from training_constants import text_to_tensor
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
token_overlap =0.3
bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

normalize_fact_familiar=1
normalize_factor=1


imgsize = 28
BP_std = 0

# helper functions:
def log_function_name():
    return inspect.stack()[1].function

def sync_devices(tensor_list):
    # uses 0th element device as the target device for all tensors in the list
    # if 2d list, will recursively sync all tensors in the list to the 0th element device of the 0th list

    if isinstance(tensor_list, list) and isinstance(tensor_list[0], list):
        target_device = tensor_list[0][0].device
        synced_tensors = [[tensor.to(target_device) for tensor in sublist] for sublist in tensor_list]
    else:
        target_device = tensor_list[0].device
        synced_tensors = [tensor.to(target_device) for tensor in tensor_list]
    return synced_tensors

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

def zero_outside_radius(tensor, center_x, center_y, radius=6):
    result = tensor.clone()
    center_x = (28*center_x)/500
    center_y = (28*center_y)/500
    
    # Iterate over all positions in the 28x28 grid
    for i in range(28):
        for j in range(28):
            # Calculate the squared distance from the current position to the center
            distance_squared = (i - center_y)**2 + (j - center_x)**2
            
            # If the distance is greater than the radius, set all channel values to zero
            if distance_squared > radius**2:
                result[:, i, j] = 0
    
    return result

def build_single(input_tensor):
    # input_tensor: [x, batch_size, channels, height, width]
    output_tensor = input_tensor.sum(dim=1)  # sum across frames
    output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)
    return output_tensor

# figures:

def simultaneous_encode(vae: VAE_CNN, data, folder_path, x):
    bpsize = 10000         #size of the binding pool
    token_overlap = 0.3
    bpPortion = int(token_overlap *bpsize)
    n = len(data)
    mse_list = []
    recon_list = []
    for i in range(n):
        activations = vae.activations(data[i].view(-1,3,28,28), False)

        BP_activations = {'l1': [activations['skip'].view(1,-1), 1]}

        # store and retrieve
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations, 1,normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations, 1,normalize_factor)
        l1_out = BP_activations_out['l1']

        recon = vae.decoder_skip_cropped(0, 0, 0, l1_out.view(1,-1))
        recon_list += [recon]
    
    save_image(torch.cat([data[:8], torch.cat(recon_list[:8])]), f'{folder_path}{x}_simultaneous.png', nrow=8, pad_value=0.6, normalize=False)
    return 0

def sequential_encode(vae: VAE_CNN, original, frames, folder_path, probe_x, input_x=6):
    bpsize = 1000         #size of the binding pool
    token_overlap = 0.3
    bpPortion = int(token_overlap *bpsize)
    # data: N,x,3,28,28  N: batch size, x: frames per item
    n = frames.size(0)
    input_x = probe_x
    mse_list = []
    recon_list = []
    probe_list = []
    for i in range(n):
        activations = vae.activations(frames[i][:probe_x].view(-1,3,28,28), True)

        BP_activations = {'shape': [activations['shape'].view(probe_x,-1), 1], 'color': [activations['color'].view(probe_x,-1), 1], } # 2 familiar in shape/color

        # store and retrieve
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations, probe_x,normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations, probe_x,normalize_factor)
        shape_out, color_out = BP_activations_out['shape'], BP_activations_out['color']        
        pre_retinal_frames = vae.decoder_cropped(activations['shape'].view(probe_x,-1), activations['color'].view(probe_x,-1),0,0)

        pre_BP_recon_frames = vae.decoder_retinal(activations['shape'].view(probe_x,-1), activations['color'].view(probe_x,-1), activations['theta'])
        pre_BP_recon_frames = F.interpolate(pre_BP_recon_frames, size=(28, 28), mode='bilinear', align_corners=False)
        
        recon_frames = vae.decoder_retinal(shape_out, color_out, activations['theta'])
        recon_frames = F.interpolate(recon_frames, size=(28, 28), mode='bilinear', align_corners=False)
        
        recon = build_single(recon_frames.view(1,probe_x,3,28,28)).view(1,3,28,28)
        probe = build_single(frames[i][:probe_x].view(1,probe_x,3,28,28)).view(1,3,28,28)
        recon_list += [recon]
        probe_list += [probe]


    save_image(torch.cat([frames[i].view(-1,3,28,28), pre_retinal_frames, pre_BP_recon_frames, recon_frames]), f'{folder_path}{probe_x}_sequential_frames.png', nrow=probe_x, pad_value=0.6, normalize=False)
    save_image(torch.cat([torch.cat(probe_list[:8]), torch.cat(recon_list[:8])]), f'{folder_path}{probe_x}_sequential.png', nrow=8, pad_value=0.6, normalize=False)

    return 0

@torch.no_grad()
def fig_simultaneous_vs_sequential_1(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()
    folder_path += '1/'

    print('generating Figure simultaneous vs sequential plot (1, 3, 6)')

    square_data_1 = torch.load(f'{DATASET_ROOT}/color_square_data/original_1_color.pth', 'cuda:1')[:40]
    square_data_3 = torch.load(f'{DATASET_ROOT}/color_square_data/original_3_color.pth', 'cuda:1')[:40]
    #square_data_3_frames = torch.load(f'{DATASET_ROOT}/color_square_data/original_frames_3_color.pth', 'cuda:1')
    square_data_6 = torch.load(f'{DATASET_ROOT}/color_square_data/original_6_color.pth', 'cuda:1')[:40]
    square_data_6_frames = torch.load(f'{DATASET_ROOT}/color_square_data/original_frames_6_color.pth', 'cuda:1')[:40]
    square_data_6_positions = torch.load(f'{DATASET_ROOT}/color_square_data/positions_6_color.pth', 'cuda:1')[:40]


    mse_data = {}
    # store 6 items simultaneously/sequentially
    mse_data['mse_simultaneous_6'] = simultaneous_encode(vae, square_data_6, folder_path, 6)

    mse_data['mse_sequential_6'] = sequential_encode(vae, square_data_6, square_data_6_frames, folder_path, 6)

    # store 3 items sequentially
    mse_data['mse_simultaneous_3'] = simultaneous_encode(vae, square_data_3, folder_path, 3)
    mse_data['mse_sequential_3'] = sequential_encode(vae, square_data_6, square_data_6_frames, folder_path, 3)

    # store 1 item
    mse_data['mse_simultaneous_1'] = simultaneous_encode(vae, square_data_1, folder_path, 1)
    mse_data['mse_sequential_1'] = sequential_encode(vae, square_data_6, square_data_6_frames, folder_path, 1)

    print(mse_data)

@torch.no_grad()
def fig_simultaneous_vs_sequential(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()
    bpsize = 10000         #size of the binding pool
    token_overlap = 0.3
    bpPortion = int(token_overlap *bpsize)


    print('generating Figure simultaneous vs sequential change detection')
    t = 200
    square_data_6 = torch.load(f'{DATASET_ROOT}/color_square_data/original_6_color.pth', 'cuda:1')[:t]
    square_data_6_frames = torch.load(f'{DATASET_ROOT}/color_square_data/original_frames_6_color.pth', 'cuda:1')[:t]
    square_data_6_change = torch.load(f'{DATASET_ROOT}/color_square_data/change_frames_6_color.pth', 'cuda:1')[:t]  # changed scenes
    square_data_6_positions = torch.load(f'{DATASET_ROOT}/color_square_data/positions_6_color.pth', 'cuda:1')[:t]

    def compute_correlation(x, y):
        x, y = x.view(-1), y.view(-1)
        x_mean, y_mean = torch.mean(x), torch.mean(y)
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sqrt(torch.sum((x - x_mean)**2) * torch.sum((y - y_mean)**2))
        return (numerator / denominator).item()

    def compute_dprime(no_change_vector, change_vector):
        hit_rate = npy.clip(npy.mean(change_vector), 0.01, 0.99)
        false_alarm_rate = npy.clip(1 - npy.mean(no_change_vector), 0.01, 0.99)
        return stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)

    def run_change_detection(recon_list, original_list, change_list, threshold):
        """
        recon_list:    list of BP-reconstructed tensors (memory)
        original_list: list of original scene tensors  (no-change probe)
        change_list:   list of changed scene tensors   (change probe)
        threshold:     scalar decision boundary
        Returns accuracy and d-prime.
        """
        recon_list, original_list, change_list = sync_devices([recon_list, original_list, change_list])
        no_change_detected, change_detected = [], []
        r_original_all, r_change_all = [], []

        for recon, orig, chng in zip(recon_list, original_list, change_list):
            r_orig = compute_correlation(recon, orig)
            r_chng = compute_correlation(recon, chng)
            r_original_all.append(r_orig)
            r_change_all.append(r_chng)

            no_change_detected.append(1 if r_orig > threshold else 0)
            change_detected.append(1 if r_chng <= threshold else 0)

        accuracy = (
            (sum(no_change_detected) / len(no_change_detected)) +
            (sum(change_detected)    / len(change_detected))
        ) / 2
        dprime = compute_dprime(no_change_detected, change_detected)
        avg_r = (npy.mean(r_original_all), npy.mean(r_change_all))
        return accuracy, dprime, avg_r

    def get_simultaneous_recons(data, frames, probe_x, positions):
        recon_list, original_list = [], []
        for i in range(len(data)):
            act = vae.activations(data[i].view(-1, 3, 28, 28), False)
            BP_act = {'l1': [act['skip'].view(1, -1), 1]}
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_act, 1, normalize_factor)
            BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_act, 1, normalize_factor)
            recon = vae.decoder_skip_cropped(0, 0, 0, BP_act_out['l1'].view(1, -1))
            recon_list.append(recon.squeeze(0))
            probe = build_single(frames[i][:probe_x].view(1, probe_x, 3, 28, 28)).squeeze(0)
            original_list.append(probe)
            #original_list.append(data[i].view(3, 28, 28))
            
        save_image(torch.cat(sync_devices([data[:8].view(8,3,28,28), torch.cat(recon_list[:8]).view(8,3,28,28)])), f'{folder_path}{probe_x}_simultaneous.png', nrow=8, pad_value=0.6, normalize=False)
    
        
        if probe_x == 1:
            for i in range(len(recon_list)):
                recon_list[i] = zero_outside_radius(recon_list[i], positions[i][0][0], positions[i][0][1], radius=6)
                #original_list[i] = zero_outside_radius(original_list[i], positions[i][0][0], positions[i][0][1], radius=6)
        return recon_list, original_list

    def get_sequential_recons(frames, probe_x, input_x=5):
        recon_list, original_list = [], []
        n = frames.size(0)
        for i in range(n):
            act = vae.activations(frames[i][:input_x].view(-1, 3, 28, 28), True)
            
            # always store all 6
            BP_act = {
                'shape': [act['shape'].view(input_x, -1), 0],
                'color': [act['color'].view(input_x, -1), 1],
            }
            BPOut, Tokenbindings = BPTokens_storage(
                bpsize, bpPortion, BP_act, input_x, normalize_factor
            )
            
            # retrieve only probe_x items
            BP_act_probe = {
                'shape': [act['shape'].view(input_x, -1)[:min(probe_x, input_x)], 0],
                'color': [act['color'].view(input_x, -1)[:min(probe_x, input_x)], 1],
            }
            BP_act_out = BPTokens_retrieveByToken(
                bpsize, bpPortion, BPOut, Tokenbindings, BP_act_probe, min(probe_x, input_x), normalize_factor
            )
            
            shape_out, color_out = BP_act_out['shape'], BP_act_out['color']
            recon_frames = vae.decoder_retinal(shape_out, color_out, act['theta'][:min(probe_x, input_x)])
            recon_frames = F.interpolate(recon_frames, size=(28, 28), mode='bilinear', align_corners=False)
            recon = build_single(recon_frames.view(1, min(probe_x, input_x), 3, 28, 28)).squeeze(0)
            probe = build_single(frames[i][:probe_x].view(1, probe_x, 3, 28, 28)).squeeze(0)
            recon_list.append(recon)
            original_list.append(probe)

        #save_image(torch.cat([frames[i].view(-1,3,28,28), pre_retinal_frames, pre_BP_recon_frames, recon_frames]), f'{folder_path}{probe_x}_sequential_frames.png', nrow=probe_x, pad_value=0.6, normalize=False)
        save_image(torch.cat(sync_devices([torch.cat(original_list[:8]).view(8,3,28,28), torch.cat(recon_list[:8]).view(8,3,28,28)])), f'{folder_path}{probe_x}_sequential.png', nrow=8, pad_value=0.6, normalize=False)

        return recon_list, original_list

    def get_change_probes(change_frames, probe_x):
        """Collapse change frames the same way get_sequential_recons collapses originals."""
        probe_list = []
        for i in range(change_frames.size(0)):
            probe = build_single(change_frames[i][:probe_x].view(1, probe_x, 3, 28, 28)).squeeze(0)
            probe_list.append(probe)
        return probe_list

    # derive a threshold from the correlation midpoint
    def midpoint_threshold(recon_list, original_list, change_list):
        recon_list, original_list, change_list = sync_devices([recon_list, original_list, change_list])
        r_orig = npy.mean([compute_correlation(r, o) for r, o in zip(recon_list, original_list)])
        r_chng = npy.mean([compute_correlation(r, c) for r, c in zip(recon_list, change_list)])
        return (r_orig + r_chng) / 2

    if not load_data:
        results = {}

        for probe_x in [1, 3, 6]:
            sim_recons, sim_originals          = get_simultaneous_recons(square_data_6, square_data_6_frames, probe_x, square_data_6_positions)
            sim_changes                        = get_change_probes(square_data_6_change, probe_x)
            sim_thresh                         = midpoint_threshold(sim_recons, sim_originals, sim_changes)
            acc, dp, avg_r                     = run_change_detection(sim_recons, sim_originals, sim_changes, sim_thresh)
            results[f'simultaneous_{probe_x}']          = {'accuracy': acc, 'dprime': dp, 'r': avg_r}

            seq_recons, seq_originals      = get_sequential_recons(square_data_6_frames, probe_x, probe_x)
            seq_changes                    = get_change_probes(square_data_6_change, probe_x)
            seq_thresh                     = midpoint_threshold(seq_recons, seq_originals, seq_changes)
            acc, dp, avg_r                 = run_change_detection(seq_recons, seq_originals, seq_changes, seq_thresh)
            results[f'sequential_{probe_x}'] = {'accuracy': acc, 'dprime': dp, 'r': avg_r}
        
        print(results)
        joblib.dump(results, pkl_path)

    else:
        results = joblib.load(pkl_path)

    mpl.rcParams['axes.titlesize']  = 16
    mpl.rcParams['axes.labelsize']  = 15
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['legend.fontsize'] = 13
    probe_sizes = [1, 3, 6]
    x = npy.arange(len(probe_sizes))
    bar_width = 0.35

    # d-prime
    fig, ax = plt.subplots(figsize=(8, 6))
    sim_dprimes = [results[f'simultaneous_{p}']['dprime'] for p in probe_sizes]
    seq_dprimes = [results[f'sequential_{p}']['dprime']   for p in probe_sizes]
    ax.bar(x - bar_width/2, sim_dprimes, width=bar_width, label='simultaneous')
    ax.bar(x + bar_width/2, seq_dprimes, width=bar_width, label='sequential')
    ax.set_xticks(x); ax.set_xticklabels(probe_sizes)
    ax.set_xlabel('probe set size')
    ax.set_ylabel("d'")
    ax.set_title("d' — simultaneous vs sequential change detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{folder_path}cd_sim_vs_seq_dprime.png')
    plt.close()

    # accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    sim_acc = [results[f'simultaneous_{p}']['accuracy'] for p in probe_sizes]
    seq_acc = [results[f'sequential_{p}']['accuracy']   for p in probe_sizes]
    ax.bar(x - bar_width/2, sim_acc, width=bar_width, label='simultaneous')
    ax.bar(x + bar_width/2, seq_acc, width=bar_width, label='sequential')
    ax.set_xticks(x); ax.set_xticklabels(probe_sizes)
    ax.set_xlabel('probe set size')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy — simultaneous vs sequential change detection')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{folder_path}cd_sim_vs_seq_accuracy.png')
    plt.close()

    # correlation — 4 bars per group: sim no-change, sim change, seq no-change, seq change
    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    gap = bar_width * 0.8
    sim_r_orig = [results[f'simultaneous_{p}']['r'][0] for p in probe_sizes]
    sim_r_chng = [results[f'simultaneous_{p}']['r'][1] for p in probe_sizes]
    seq_r_orig = [results[f'sequential_{p}']['r'][0]   for p in probe_sizes]
    seq_r_chng = [results[f'sequential_{p}']['r'][1]   for p in probe_sizes]
    ax.bar(x + offsets[0]*gap, sim_r_orig, width=bar_width*0.8, label='sim — no change')
    ax.bar(x + offsets[1]*gap, sim_r_chng, width=bar_width*0.8, label='sim — change')
    ax.bar(x + offsets[2]*gap, seq_r_orig, width=bar_width*0.8, label='seq — no change')
    ax.bar(x + offsets[3]*gap, seq_r_chng, width=bar_width*0.8, label='seq — change')
    ax.set_xticks(x); ax.set_xticklabels(probe_sizes)
    ax.set_xlabel('probe set size')
    ax.set_ylabel('r')
    ax.set_title('Correlation — simultaneous vs sequential')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{folder_path}cd_sim_vs_seq_r.png')
    plt.close()

def batch_ssim(x, y):
    """x, y: (B, C, H, W) torch tensors. Returns mean SSIM across batch."""
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    vals = []
    for i in range(x_np.shape[0]):
        # (C, H, W) -> (H, W, C)
        s = ssim(x_np[i].transpose(1,2,0), y_np[i].transpose(1,2,0),
                 data_range=1.0, channel_axis=-1)
        vals.append(s)
    return sum(vals) / len(vals)

@torch.no_grad()
def fig_repeat_recon(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    vae.eval()
    print('generating Figure repeated reconstructions, green 5, red 5, red 3')
    retina_size = 100
    imgsize = 28
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    numimg = 3  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    test_loader_noSkip = Dataset('emnist',{'colorize':False}, train=True).get_loader(numimg)

    dataiter_noSkip = iter(test_loader_noSkip)
    data, labels = next(dataiter_noSkip)
    data = data #.cuda()

    # find 2 5's and then 1 3
    imgs = []
    c = 0
    while c < 2:
        if labels[0][0].item() == 11:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    c = 0
    while c < 1:
        if labels[0][0].item() == 12:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    red = Colorize_specific(0)
    
    green = Colorize_specific(1)
    blue = Colorize_specific(2)

    imgs[0] = convert_tensor(red(convert_image(imgs[0])))
    imgs[1] = convert_tensor(green(convert_image(imgs[1])))
    imgs[2] = convert_tensor(blue(convert_image(imgs[2])))

    sample = torch.cat(imgs, 0).cuda()
    
    #push the images through the model
    activations = vae.activations(sample.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']
    reconb = vae.decoder_cropped(shape_act, color_act, 0)

    mu_shape, _, mu_color, _, hskip = vae.encoder(sample.view(-1,3,28,28))

    #    sample = torch.cat(imgs, 0).cuda()
    #sample = torch.stack(imgs, 0).cuda()
    #with torch.no_grad():
    #    reconb, _, _, _, _, _, _ = vae(sample, 'cropped', ['shape', 'color'])
    

    reconskip = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(numimg,-1))
    #reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample.view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample.view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act

    BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}
    
    emptyshape = torch.zeros((1,3,28,28)).cuda()
    
    # Row 1: originals (already in imgmatrixMap and imgmatrixL1)
    # Row 2: direct recon (already in imgmatrixMap and imgmatrixL1)
    
    # Rows 3+: BP recon at set sizes 1 through numimg
    for n in range(1, numimg+1):
        # Store and retrieve shape+color maps
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n, normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n, normalize_factor)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all, 0, 0).cuda()
        
        # Store and retrieve L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n, normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n, normalize_factor)
        l1_out_all = BP_activations_out['l1']
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n, -1))
        
        # Append retrievals, pad with empty images for items not stored
        imgmatrixMap = torch.cat([imgmatrixMap, retrievals], 0)
        imgmatrixL1 = torch.cat([imgmatrixL1, recon_layer1_skip], 0)
        
        for i in range(n, numimg):
            imgmatrixMap = torch.cat([imgmatrixMap, emptyshape], 0)
            imgmatrixL1 = torch.cat([imgmatrixL1, emptyshape], 0)
    
    save_image(imgmatrixL1, f'{folder_path}figure_repeat_L1.png', nrow=numimg, normalize=False, pad_value=0.6)
    save_image(imgmatrixMap, f'{folder_path}figure_repeat_Map.png', nrow=numimg, normalize=False, pad_value=0.6)


@torch.no_grad()
def fig_non_repeat_recon(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    vae.eval()
    print('generating Figure repeated reconstructions, green 5, red 5, red 3')
    retina_size = 100
    imgsize = 28
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    numimg = 3  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    test_loader_noSkip= Dataset('emnist',{'colorize':False}, train=True).get_loader(numimg)

    dataiter_noSkip = iter(test_loader_noSkip)
    data, labels = next(dataiter_noSkip)
    data = data #.cuda()

    # find 2 5's and then 1 3
    imgs = []
    c = 0
    while c < 1:
        if labels[0][0].item() == 10:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    c = 0
    while c < 1:
        if labels[0][0].item() == 11:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    c = 0
    while c < 1:
        if labels[0][0].item() == 12:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)

    red = Colorize_specific(0)
    
    green = Colorize_specific(1)
    blue = Colorize_specific(2)

    imgs[0] = convert_tensor(red(convert_image(imgs[0])))
    imgs[1] = convert_tensor(green(convert_image(imgs[1])))
    imgs[2] = convert_tensor(blue(convert_image(imgs[2])))

    sample = torch.cat(imgs, 0).cuda()
    
    #push the images through the model
    activations = vae.activations(sample.view(-1,3,28,28), False)
    l1_act = activations['skip']
    shape_act = activations['shape']
    color_act = activations['color']
    reconb = vae.decoder_cropped(shape_act, color_act, 0)

    mu_shape, _, mu_color, _, hskip = vae.encoder(sample.view(-1,3,28,28))

    #    sample = torch.cat(imgs, 0).cuda()
    #sample = torch.stack(imgs, 0).cuda()
    #with torch.no_grad():
    #    reconb, _, _, _, _, _, _ = vae(sample, 'cropped', ['shape', 'color'])
    

    reconskip = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(numimg,-1))
    #reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample.view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample.view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act

    BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1], 'color': [color_act.view(numimg,-1), 1]}
    BP_activations_l1 = {'l1': [l1_act.view(numimg,-1), 1]}
    
    emptyshape = torch.zeros((1,3,28,28)).cuda()
    
    # Row 1: originals (already in imgmatrixMap and imgmatrixL1)
    # Row 2: direct recon (already in imgmatrixMap and imgmatrixL1)
    
    # Rows 3+: BP recon at set sizes 1 through numimg
    for n in range(1, numimg+1):
        # Store and retrieve shape+color maps
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n, normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n, normalize_factor)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all, 0, 0).cuda()
        
        # Store and retrieve L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n, normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n, normalize_factor)
        l1_out_all = BP_activations_out['l1']
        recon_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, l1_out_all.view(n, -1))
        
        # Append retrievals, pad with empty images for items not stored
        imgmatrixMap = torch.cat([imgmatrixMap, retrievals], 0)
        imgmatrixL1 = torch.cat([imgmatrixL1, recon_layer1_skip], 0)
        
        for i in range(n, numimg):
            imgmatrixMap = torch.cat([imgmatrixMap, emptyshape], 0)
            imgmatrixL1 = torch.cat([imgmatrixL1, emptyshape], 0)
    
    save_image(imgmatrixL1, f'{folder_path}figure_non_repeat_L1.png', nrow=numimg, normalize=False, pad_value=0.6)
    save_image(imgmatrixMap, f'{folder_path}figure_non_repeat_Map.png', nrow=numimg, normalize=False, pad_value=0.6)

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
    test_loader_noSkip= Dataset('emnist',{'colorize':False}, train=True).get_loader(numimg)  

    dataiter_noSkip = iter(test_loader_noSkip)
    data, labels = next(dataiter_noSkip)
    data = data #.cuda()

    # find 2 5's and then 1 3
    imgs = []
    c = 0
    while c < 2:
        if labels[0][0].item() == 15:
            imgs += [data[0]]
            c += 1
        data, labels = next(dataiter_noSkip)
    
    c = 0
    while c < 1:
        if labels[0][0].item() == 13:
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
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, n,normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, n,normalize_factor)
        shape_out_all, color_out_all = BP_activations_out['shape'], BP_activations_out['color']
        z = torch.randn(numimg-n,8).cuda()
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()
        #retrievals = retrievals[:n]
        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, n,normalize_factor)
        BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, n,normalize_factor)
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
@torch.no_grad()

def percept_concept(vae: VAE_CNN, shape_label, s_classes, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    mnist_iter = iter(Dataset('mnist',{'retina':False, 'colorize':True}, train=True).get_loader(1))
    emnist_iter = iter(Dataset('emnist', {'retina':False, 'colorize':True}, train=True).get_loader(1))

    vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


    pairs = [
        (6,  24),  # 6+O → forms a rounder 8-like shape
        (21, 15),  # F+L → E  (F has top/mid bars, L adds the bottom bar)
        (9,  6),   # 6+9 → 8  (top circle + bottom circle)
    ]

    pairs = [
        (6,  24,  0.0,  0.0, 1.0),   # digit 3 + concept 8,  centered
        (21, 15,  0.3,  0.0, 1.0),   # S + concept 8,         shift right
        (6, 9, 0.0,  0.2, 0.9),   # P + concept A,         shift down, slightly smaller
    ]
    vae.eval()
    #loop through the pairs
    for percept_label, label, tx, ty, s  in pairs:
        if percept_label >= 10:
            loader = Dataset('emnist', {'retina': False, 'colorize': True, 'target_set': [percept_label]}, train=True).get_loader(1)
        else:
            loader = Dataset('mnist',  {'retina': False, 'colorize': True, 'target_set': [percept_label]}, train=True).get_loader(1)
        #show the model a visual shape, get the latent
        img = next(iter(loader))[0][0].to(device).view(1, 3, 28, 28)
        activations    = vae.activations(img, False)
        z_shape_img    = activations['shape']
        z_color_img    = activations['color']

        #now generate a different object with a label
        onehot_label = F.one_hot(torch.tensor([label]).to(device), num_classes=s_classes).float()
        z_shape_labels = shape_label(onehot_label, n=1)
        combined       = (z_shape_labels + z_shape_img) * 0.5

        percept_name = emnist_labels[percept_label]
        concept_name = emnist_labels[label]

        save_image(vae.decoder_cropped(combined,       z_color_img), f'{folder_path}percept_concept_{percept_name}-{concept_name}.png')
        save_image(vae.decoder_cropped(z_shape_labels, z_color_img), f'{folder_path}percept_concept_{concept_name}_label.png')
        save_image(img, f'{folder_path}percept_{percept_name}.png')
    # --- all identities reconstructed from labels only ---
    label_recons = []
    for lbl in range(s_classes):
        onehot = F.one_hot(torch.tensor([lbl]).to(device), num_classes=s_classes).float()
        z = shape_label(onehot, n=1)
        label_recons.append(vae.decoder_shape(z))

    save_image(torch.cat(label_recons, 0),
               f'{folder_path}label_recons_all.png',
               nrow=s_classes, normalize=False, pad_value=0.6)

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
    #bpsize = 10000#00         #size of the binding pool
    #token_overlap =0.3
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

    pre_bp = vae.decoder_skip_cropped(0, 0, 0, l1_act.view(1,-1))

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
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_l1, 1,normalize_factor)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_l1, 1,normalize_factor)
        BP_layerI_out = BP_act_out['l1']

        BP_layer1_skip = vae.decoder_skip_cropped(0, 0, 0, BP_layerI_out.view(1,-1))
        # add some gaussian blur to eliminate the stippling
        BP_layer1_skip = Ft.gaussian_blur(BP_layer1_skip, kernel_size=[3, 3], sigma=[500.0, 500.0])

        # reconstruct  from BP version of layer 1, run through the bottleneck
        bn_act = vae.activations(0, False, BP_layerI_out.view(1,-1))
        BP_layer1_noskip = vae.decoder_cropped(bn_act['shape'], bn_act['color'])
        
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, 1,normalize_factor)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, 1,normalize_factor)
        shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
        #reconstruct from BP version of the shape and color maps
        retrievals = vae.decoder_cropped(shape_out_BP, color_out_BP,0,0).cuda()

        imgmatrixL1skip = torch.cat([imgmatrixL1skip,BP_layer1_skip])
        imgmatrixL1noskip = torch.cat([imgmatrixL1noskip,BP_layer1_noskip])
        imgmatrixMap= torch.cat([imgmatrixMap,retrievals])
    
    store2 = []
    #save an image showing:  original images, reconstructions directly from L1,  from L1 BP, from L1 BP through bottleneck, from maps BP
    save_image(torch.cat([imgs[0: numimg].view(numimg, 3, 28, imgsize), pre_bp, imgmatrixL1skip, imgmatrixL1noskip, imgmatrixMap], 0), f'{folder_path}fig_novel_rep.png',
            nrow=numimg,            normalize=False,)


@torch.no_grad()
#take a stimulus and make a series of modifications
@torch.no_grad()
def fig_retinal_mod(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    vae.eval()
    bs = 10
    #generate a random object for each column in the figure and extract its features
    emnist_loader = Dataset('emnist', {'retina': True, 'colorize': True, 'scale': True,
                                       'target_set': list(range(10, 36))}, train=True).get_loader(bs)
    dataiter = iter(emnist_loader)
    data, _ = next(dataiter)
    data = data[0].cuda()

    act      = vae.activations(data, True)
    shape    = act['shape']
    color    = act['color']
    scale    = act['scale']
    location = act['location']
    theta    = torch.cat([scale, location], 1)

    def make_theta(s=None, loc=None):
        s   = s   if s   is not None else scale
        loc = loc if loc is not None else location
        return torch.cat([s, loc], 1)

    # scale sweep: smaller → larger parametric 
    scale_rows = []
    for sv in [0.5, 1.3]:
        s = torch.full((bs, 1), sv).cuda()
        scale_rows.append(vae.decoder_retinal(shape, color, make_theta(s=s)))

    # location sweep: left → right (y held at 0)
    loc_rows = []
    for lx in [-0.6, 0.6]:
        loc = torch.tensor([lx, 0.0] * bs).view(bs, 2).cuda()
        loc_rows.append(vae.decoder_retinal(shape, color, make_theta(loc=loc)))

    # color sweep: encode a fresh batch, broadcast each color across all items 
    data2, _ = next(dataiter)
    color_samples = vae.activations(data2[0].cuda(), True)['color']
    color_rows = []
    for i in range(2):
        c = color_samples[i].unsqueeze(0).expand(bs, -1)
        color_rows.append(vae.decoder_retinal(shape, c, theta))

    all_rows = (
        [data,                                     # row 0: originals
         vae.decoder_retinal(shape, color, theta)] # row 1: identity recon
        + scale_rows                               # rows 2-6: scale sweep
        + loc_rows                                 # rows 7-11: location sweep
        + color_rows                               # rows 12-16: color sweep
    )

    save_image(torch.cat(all_rows, 0),
               f'{folder_path}figure_retinal_mod.png',
               nrow=bs, normalize=False, pad_value=0.6)

    
@torch.no_grad()
def recon_test(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()
    bpsize = 10000#00         #size of the binding pool
    token_overlap =0.3
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

    bs = 10
    device = next(vae.parameters()).device
    
    obj_1_transforms = {'retina':True, 'colorize':True, 'scale':False, 'target_set': quickdraw_target_set,  'location_targets':{(-1,-1):list(range(0,s_classes+1))}}
    obj_1_loader = Dataset('quickdraw', obj_1_transforms).get_loader(bs)

    data, labels = next(iter(obj_1_loader))

    activations = vae.activations(data[0].to(device), True)

    shape_recon = vae.decoder_shape(activations['shape'], 0, 0)
    color_recon = vae.decoder_color(0, activations['color'], 0)
    object_recon = vae.decoder_object(activations['object'], 0, 0)
    skip_recon = vae.decoder_skip_cropped(0, 0, 0, activations['skip'])
    save_image(
        torch.cat([data[1].to(device), skip_recon, object_recon, color_recon, shape_recon], 0),
        f'{folder_path}figure_recon_test.png', pad_value=0.6,
        nrow=bs, normalize=False)

def composite_scene(sample_1, sample_2, sample_3, device):
    def get_mask(img, threshold=0.015):
        img.squeeze_(0)
        r, g, b = img[0:1], img[1:2], img[2:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (luminance > threshold).float()

    comb_img = sample_1.to(device)
    for fg in (sample_2.to(device), sample_3.to(device)):
        mask = get_mask(fg)
        comb_img = comb_img * (1 - mask) + fg * mask

    return torch.clamp(comb_img, 0, 1)

def find_closest_index(val_list, values):
    value = val_list.tolist()
    out = []
    for value in val_list:
        if value in values:
            v = value
        else:
            v = min(values, key=lambda x: (abs(x - value), x))
        out += [v]
    return torch.tensor(out)

def obj_scene_helper(vae, dataloaders, n, save_img, return_loss, bpsize, bpPortion, object_label, color_label, object_classifier, color_classifier, folder_path):
    comb_img_col = []
    obj_recon_no_BP_col = []
    obj_recon_BP_col = []
    obj_recon_label_col = []
    obj_recon_BP_label_col = []
    holistic_no_BP_col = []
    holistic_BP_col = []
    hybrid_col = []
    hybrid_1hot_col = []
    hybrid_bitmask_col = []
    obj_1_loader, obj_2_loader, obj_3_loader, bs = dataloaders
    
    for scene_idx in range(n):
        data_1, labels_1 = next(iter(obj_1_loader))
        data_2, labels = next(iter(obj_2_loader))
        data_3, labels = next(iter(obj_3_loader))

        # add the images together to form a scene
        comb_img = composite_scene(data_1[2], data_2[0], data_3[0], device)
        holistic_comb_img = F.interpolate(comb_img, size=(28, 28), mode='bilinear', align_corners=False)

        activations_1_r = vae.activations(data_1[0], True, None, 'object')
        activations_2_r = vae.activations(data_2[0], True, None, 'object')
        activations_3_r = vae.activations(data_3[0], True, None, 'object')
        activations_1 = vae.activations(data_1[1], False, None, 'object')
        activations_2 = vae.activations(data_2[1], False, None, 'object')
        activations_3 = vae.activations(data_3[1], False, None, 'object')
        activations_1['theta'] = activations_1_r['theta']
        activations_2['theta'] = activations_2_r['theta']
        activations_3['theta'] = activations_3_r['theta']

        
        holistict_activations = vae.activations(holistic_comb_img, False, None, None)
        l1_activations_1 = vae.activations(data_1[1], False, None, None)
        l1_act_item1 = l1_activations_1['skip']

        obj_1, color_1, theta_1 = activations_1['object'], activations_1['color'], activations_1['theta']
        obj_2, color_2, theta_2 = activations_2['object'], activations_2['color'], activations_2['theta']
        obj_3, color_3, theta_3 = activations_3['object'], activations_3['color'], activations_3['theta']

        objs = torch.cat([obj_1, obj_2, obj_3], 0)
        colors = torch.cat([color_1, color_2, color_3], 0)
        thetas = torch.cat([theta_1, theta_2, theta_3], 0)

        BP_activations = {'object': [objs.view(3, -1), 1],
                        'color': [colors.view(3, -1), 1]}

        # now store/retrieve from object and color maps
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations, 3, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations, 3, normalize_factor)
        BP_obj_out = BP_act_out['object']
        BP_color_out = BP_act_out['color']

        # now store/retrieve via object/color classifications to 1-hot
        pred_objs = object_classifier.predict(objs.cpu())
        pred_colors = color_classifier.predict(colors.cpu())

        object_oneHot = F.one_hot(torch.tensor(pred_objs), num_classes=s_classes).float().to(device)
        color_oneHot = F.one_hot(torch.tensor(pred_colors), num_classes=10).float().to(device)

        BP_activations_oneHot = {'object': [object_oneHot.view(3, -1), 1],
                                'color': [color_oneHot.view(3, -1), 1]}

        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_oneHot, 3, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_oneHot, 3, normalize_factor)
        BP_obj_out_label = BP_act_out['object']
        BP_color_out_label = BP_act_out['color']

        # remove noise from BP output by converting back to 1-hot
        BP_obj_indices = torch.argmax(BP_obj_out_label, dim=-1)
        BP_color_indices = torch.argmax(BP_color_out_label, dim=-1)
        BP_obj_indices = find_closest_index(BP_obj_indices, [0, 2, 8, 10, 11])
        BP_obj_out_label = F.one_hot(BP_obj_indices, num_classes=BP_obj_out_label.size(-1)).to(device).float()
        BP_color_out_label = F.one_hot(BP_color_indices, num_classes=BP_color_out_label.size(-1)).to(device).float()

        obj_from_BP_label = object_label(BP_obj_out_label, 1)
        color_from_BP_label = color_label(BP_color_out_label, 1)
        obj_from_label = object_label(object_oneHot, 1)
        color_from_label = color_label(color_oneHot, 1)

        # holistic BP
        BP_activations = {'l1': [holistict_activations['skip'], 1]}

        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations, 1, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations, 1, normalize_factor)
        BP_l1_out = BP_act_out['l1']

        # hybrid holistic bg + object_latent:
        BP_activations = {'object': [objs.view(3, -1), 1],
                            'color': [colors.view(3, -1), 1],
                            'l1': [holistict_activations['skip'].view(3,-1), 1]}
        
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations, 3, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations, 3, normalize_factor)
        BP_l1_H = BP_act_out['l1']
        BP_object_H = BP_act_out['object']
        BP_color_H = BP_act_out['color']

        obj_recon_H = vae.decoder_retinal_object(BP_object_H, BP_color_H, thetas, 0).sum(dim=0, keepdim=True)
        holistic_H = vae.decoder_skip_cropped(0, 0, 0, BP_l1_H)
        holistic_H = F.interpolate(holistic_H, size=(64, 64), mode='bilinear', align_corners=False)
        holistic_H = Ft.gaussian_blur(holistic_H, kernel_size=[79, 79], sigma=[5.5, 5.5]) #F.avg_pool2d(holistic_H, kernel_size=15, stride=1, padding=15 // 2)
        hybrid = composite_scene(holistic_H, obj_recon_H[0], torch.zeros_like(obj_recon_H[0]), device)
        hybrid_col.append(hybrid.view(bs, 3, 64, 64))

        # hybrid holistic bg + object one hots:
        BP_activations_oneHot_H = {'object': [object_oneHot.view(3, -1), 1],
                                'color': [color_oneHot.view(3, -1), 1],
                                'l1': [holistict_activations['skip'].view(3,-1), 1]}       
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_oneHot_H, 3, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_oneHot_H, 3, normalize_factor)
        BP_obj_out_label = BP_act_out['object']
        BP_color_out_label = BP_act_out['color']
        BP_l1_H = BP_act_out['l1']

        # remove noise from BP output by converting back to 1-hot
        BP_obj_indices = torch.argmax(BP_obj_out_label, dim=-1)
        BP_color_indices = torch.argmax(BP_color_out_label, dim=-1)
        BP_obj_indices = find_closest_index(BP_obj_indices, [0, 2, 8, 10, 11])
        BP_obj_out_label = F.one_hot(BP_obj_indices, num_classes=BP_obj_out_label.size(-1)).to(device).float()
        BP_color_out_label = F.one_hot(BP_color_indices, num_classes=BP_color_out_label.size(-1)).to(device).float()

        obj_from_BP_label_H = object_label(BP_obj_out_label, 1)
        color_from_BP_label_H = color_label(BP_color_out_label, 1)

        obj_recon_H = vae.decoder_retinal_object(obj_from_BP_label_H, color_from_BP_label_H, thetas, 0).sum(dim=0, keepdim=True)
        holistic_H = vae.decoder_skip_cropped(0, 0, 0, BP_l1_H)
        holistic_H = F.interpolate(holistic_H, size=(64, 64), mode='bilinear', align_corners=False)
        holistic_H = Ft.gaussian_blur(holistic_H, kernel_size=[79, 79], sigma=[5.5, 5.5])
        hybrid_H = composite_scene(holistic_H, obj_recon_H[0], torch.zeros_like(obj_recon_H[0]), device)
        hybrid_1hot_col.append(hybrid_H.view(bs, 3, 64, 64))

        # hybrid holistic bg + 2 QuickDraw one-hots + 1 fashion-MNIST L1 skip:
        BP_activations_oneHot_H = {'l1_item': [l1_act_item1.view(1, -1), 1],
                                'object_1hot': [object_oneHot[1:].view(2, -1), 1],
                                'color_1hot': [color_oneHot[1:].view(2, -1), 1],
                                'l1': [holistict_activations['skip'].view(1,-1), 1],
                                'act_bitmask': [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
                                'act_name_map': ['l1_item', 'object_1hot', 'color_1hot', 'l1']}       
        BPOut, Tokenbindings = BPTokens_storage_bitmask(bpsize, bpPortion, BP_activations_oneHot_H, 4, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken_bitmask(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_oneHot_H, 4, normalize_factor)
        BP_l1_item = BP_act_out['l1_item']
        BP_obj_out_label = BP_act_out['object_1hot']
        BP_color_out_label = BP_act_out['color_1hot']
        BP_l1_H = BP_act_out['l1']

        # snap the 2 QuickDraw one-hots back to valid classes
        BP_obj_indices = torch.argmax(BP_obj_out_label, dim=-1)
        BP_color_indices = torch.argmax(BP_color_out_label, dim=-1)
        BP_obj_indices = find_closest_index(BP_obj_indices, [0, 2, 8, 10, 11])
        BP_obj_out_label = F.one_hot(BP_obj_indices, num_classes=BP_obj_out_label.size(-1)).to(device).float()
        BP_color_out_label = F.one_hot(BP_color_indices, num_classes=BP_color_out_label.size(-1)).to(device).float()

        obj_from_BP_label_H = object_label(BP_obj_out_label, 1)
        color_from_BP_label_H = color_label(BP_color_out_label, 1)

        # reconstruct QuickDraw objects from one-hot at their retinal positions
        onehot_recon = vae.decoder_retinal_object(obj_from_BP_label_H, color_from_BP_label_H, thetas[1:], 0).sum(dim=0, keepdim=True)

        # reconstruct fashion_MNIST item from L1 skip (28x28 crop), place at its retinal position
        fashion_mnist_recon = vae.decoder_skip_cropped(0, 0, 0, BP_l1_item)
        fashion_mnist_recon = Ft.gaussian_blur(fashion_mnist_recon , kernel_size=[3, 3], sigma=[1, 1])

        fashion_mnist_retinal = torch.zeros(1, 3, 64, 64).to(device)
        pad_left = int(labels_1[2].item() - 14)
        pad_top = int(labels_1[3].item() - 14)
        pad_left = max(0, min(pad_left, 36))  # clamp to valid range (64-28=36)
        pad_top = max(0, min(pad_top, 36))
        fashion_mnist_retinal[:, :, pad_top:pad_top+28, pad_left:pad_left+28] = fashion_mnist_recon

        # holistic background
        holistic_H = vae.decoder_skip_cropped(0, 0, 0, BP_l1_H)
        holistic_H = F.interpolate(holistic_H.view(1, 3, 28, 28), size=(64, 64), mode='bilinear', align_corners=False)
        holistic_H = Ft.gaussian_blur(holistic_H, kernel_size=[79, 79], sigma=[5.5, 5.5])

        # composite: holistic bg → fashion_MNIST L1 → QuickDraw one-hot objects on top
        hybrid_H = composite_scene(holistic_H, fashion_mnist_retinal[0], torch.zeros_like(fashion_mnist_retinal[0]), device)
        hybrid_H = composite_scene(hybrid_H, onehot_recon[0], torch.zeros_like(onehot_recon[0]), device)
        hybrid_bitmask_col.append(hybrid_H.view(bs, 3, 64, 64))


        # reconstruct
        obj_recon_no_BP = vae.decoder_retinal_object(objs, colors, thetas, 0).sum(dim=0, keepdim=True)
        obj_recon_BP = vae.decoder_retinal_object(BP_obj_out, BP_color_out, thetas, 0).sum(dim=0, keepdim=True)
        obj_recon_BP_label = vae.decoder_retinal_object(obj_from_BP_label, color_from_BP_label, thetas, 0).sum(dim=0, keepdim=True)
        obj_recon_label = vae.decoder_retinal_object(obj_from_label, color_from_label, thetas, 0).sum(dim=0, keepdim=True)
        holistic_BP = vae.decoder_skip_cropped(0, 0, 0, BP_l1_out)
        holistic_no_BP = vae.decoder_skip_cropped(0, 0, 0, holistict_activations['skip'])

        #holistic_BP = Ft.gaussian_blur(holistic_BP, kernel_size=[3, 3], sigma=[1, 1])
        #holistic_no_BP = Ft.gaussian_blur(holistic_no_BP, kernel_size=[3, 3], sigma=[1, 1])

        holistic_BP = F.interpolate(holistic_BP, size=(64, 64), mode='bilinear', align_corners=False)
        holistic_no_BP = F.interpolate(holistic_no_BP, size=(64, 64), mode='bilinear', align_corners=False)


        comb_img_col.append(comb_img.view(bs, 3, 64, 64))
        obj_recon_no_BP_col.append(obj_recon_no_BP.view(bs, 3, 64, 64))
        obj_recon_BP_col.append(obj_recon_BP.view(bs, 3, 64, 64))
        obj_recon_label_col.append(obj_recon_label.view(bs, 3, 64, 64))
        obj_recon_BP_label_col.append(obj_recon_BP_label.view(bs, 3, 64, 64))
        holistic_no_BP_col.append(holistic_no_BP.view(bs, 3, 64, 64))
        holistic_BP_col.append(holistic_BP.view(bs, 3, 64, 64))

    if save_img:
            from torchvision.utils import make_grid
            
            row_labels = [
                'Original scene',
                'Object latents (no BP)',
                'Object latents (BP)',
                'Object from labels (no BP)',
                'Object from labels (BP)',
                'Holistic L1 (no BP)',
                'Holistic L1 (BP)',
                'Hybrid: latent obj + hol bg',
                'Hybrid: 1-hot obj + hol bg',
                'Hybrid: bitmask obj + hol bg',
            ]
            
            rows = [torch.cat(comb_img_col, 0), torch.cat(obj_recon_no_BP_col, 0),
                    torch.cat(obj_recon_BP_col, 0), torch.cat(obj_recon_label_col, 0),
                    torch.cat(obj_recon_BP_label_col, 0), torch.cat(holistic_no_BP_col, 0),
                    torch.cat(holistic_BP_col, 0), torch.cat(hybrid_col, 0),
                    torch.cat(hybrid_1hot_col, 0), torch.cat(hybrid_bitmask_col, 0)]
            
            grid = make_grid(torch.cat(rows, 0), nrow=n, normalize=False, pad_value=0.6)
            grid_img = convert_image(grid.cpu())
            
            label_width = 200
            new_width = label_width + grid_img.width
            labeled_img = Image.new('RGB', (new_width, grid_img.height), (153, 153, 153))
            labeled_img.paste(grid_img, (label_width, 0))
            
            draw = ImageDraw.Draw(labeled_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            row_height = grid_img.height / len(row_labels)
            for i, label in enumerate(row_labels):
                y = int(i * row_height + row_height / 2 - 7)
                draw.text((5, y), label, fill=(0, 0, 0), font=font)
            
            labeled_img.save(f'{folder_path}figure_obj_scene_{scene_idx}.png')


    if return_loss:
        orig = torch.cat(comb_img_col, 0)
        brightness = orig.max(dim=1, keepdim=True).values
        fg_mask = (brightness > 0.01).float()


        losses = {
            'holistic_no_BP': batch_ssim(torch.cat(holistic_no_BP_col, 0),orig),
            'holistic_BP': batch_ssim(torch.cat(holistic_BP_col, 0),orig),
            'latent_obj_holistic_bg': batch_ssim(torch.cat(hybrid_col, 0),orig),
            '1hot_obj_holistic_bg': batch_ssim(torch.cat(hybrid_1hot_col, 0),orig),
            'hybrid_obj': batch_ssim(torch.cat(hybrid_bitmask_col, 0),orig)}
        return losses

def batch_ssim(pred, target, data_range=1.0):
    p = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    t = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    scores = [ssim(t[i], p[i], data_range=data_range, channel_axis=-1)
              for i in range(p.shape[0])]
    return sum(scores) / len(scores)


@torch.no_grad()
def fig_obj_scene_recon(vae: VAE_CNN, object_label, color_label, object_classifier, color_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()
    bpsize = 10000#00         #size of the binding pool
    token_overlap =0.3
    n = 7
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

    bs = 1
    device = next(vae.parameters()).device
    object_label.to(device)
    # holistic rep for background, lowpass filter average, latent representation for objects, reconstruct and layer
    # SSIM for each reconstruction row over 1000 trials.
    quickdraw_target_set = [0, 2, 8, 10, 11]
    #novel stimulus
    obj_1_transforms = {'retina': True, 'colorize': True, 'scale': False, 'target_set': [37],
                        'location_targets': {(-1, -1): list(range(0, s_classes + 1))}, 'colorize_background':'split'}
    obj_1_loader = Dataset('fashion_mnist', obj_1_transforms).get_loader(bs)

    obj_2_transforms = {'retina': True, 'colorize': True, 'scale': False, 'target_set': quickdraw_target_set,
                        'location_targets': {(-1, 1): list(range(0, s_classes + 1))}}
    obj_2_loader = Dataset('quickdraw', obj_2_transforms).get_loader(bs)

    obj_3_transforms = {'retina': True, 'colorize': True, 'scale': False, 'target_set': quickdraw_target_set,
                        'location_targets': {(1, 1): list(range(0, s_classes + 1))}}
    obj_3_loader = Dataset('quickdraw', obj_3_transforms).get_loader(bs)

    #novel stimulus
    obj_1_transforms = {'retina': True, 'colorize': True, 'scale': False, 'target_set': [37],
                        'location_targets': {(-1, -1): [37]}, 'colorize_background':'split'}
    #obj_1_loader = Dataset('mnist', obj_1_transforms).get_loader(bs)
    obj_1_loader = Dataset('fashion_mnist', obj_1_transforms).get_loader(bs)

    dataloaders = (obj_1_loader, obj_2_loader, obj_3_loader, bs)

    losses_junk = obj_scene_helper(vae, dataloaders, n, save_img=True, return_loss=True, bpsize=bpsize, bpPortion=bpPortion, object_label=object_label, color_label=color_label, object_classifier=object_classifier, color_classifier=color_classifier, folder_path=folder_path)
    loss_trials = 100
    losses_dict = {}
#    for bpsize in [15000, 10000, 7000, 5000, 4000, 3000, 1000, 500]:
    for bpsize in [10000]:
        losses_dict[bpsize] = obj_scene_helper(vae, dataloaders, loss_trials, save_img=False, return_loss=True, bpsize=bpsize, bpPortion=bpPortion, object_label=object_label, color_label=color_label, object_classifier=object_classifier, color_classifier=color_classifier, folder_path=folder_path)


    for bps in losses_dict:
        losses = losses_dict[bps]
        names = list(losses.keys())
        values = list(losses.values())
        # ingredient vector mark token as shape, color, etc

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, values, color='steelblue')

        ax.set_ylabel('SSIM score')
        ax.set_title(f'SSIM score vs Reconstruction Method {loss_trials} Trials, BP size: {bps}')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{folder_path}reconstruction_error_{bps}.png')

def visual_synthesis_helper(vae: VAE_CNN, num1, num2, theta, bs, shape_label: VAEshapelabels, s_classes, object_classifier, folder_path: str, save_images: bool = False):
    letter1 = emnist_labels[num1]
    letter2 = emnist_labels[num2]
    device = next(vae.parameters()).device
    shape_label.to(device)
    # 1. generate shape latents from one-hot labels through the label network
    num_labels = F.one_hot(torch.tensor([num1, num2]).to(device), num_classes=s_classes).float().to(device) # shape
    z_shape = shape_label(num_labels, 10)

    # 2. decode each letter onto the retina with its specific theta
    recon_crop = vae.decoder_shape(z_shape)    
    recon = vae.decoder_retinal(z_shape, 0, theta, 'shape')
    img1 = recon[0]
    img2 = recon[1]
    
    # 3. add the two retinal images together
    comb_img = torch.clamp(img1 + img2, 0, 0.5) * 1.3
    # 4. pass the composite back through the encoder as a retinal image
    comb_img = comb_img.view(1,3,64,64)

    activations = vae.activations(comb_img, True, None, 'object')

    # 5. classify the object latent
    pred_ss = object_classifier.predict(activations['object'].cpu())
    out_pred = pred_ss[0]    
    recon_object = vae.decoder_retinal_object(activations['object'], activations['color'], activations['theta'], 0)
    if save_images:
#        recon_shape_retinal = vae.decoder_retinal_object(activations['object'], activations['color'], activations['theta'], 0)
        save_image(comb_img, f'{folder_path}{letter1}_{letter2}_assembledobject.png')
        save_image(recon_object, f'{folder_path}{letter1}_{letter2}_reconstructedobject.png')
        save_image(recon_crop, f'{folder_path}{letter1}_{letter2}_pairfromLabelNetwork.png')
        save_image(img1, f'{folder_path}{letter1}.png')
        save_image(img2, f'{folder_path}{letter2}.png')
        save_image(activations['stn_out'], f'{folder_path}stn_out.png')
    return out_pred, comb_img.squeeze(0)

@torch.no_grad()
def fig_visual_synthesis_umbrella(vae: VAE_CNN, shape_label, s_classes, object_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    vae.eval()
    bs = 2
    num1 = 13 # D
    num2 = 19 # J

    location = torch.tensor([[0.35, 0.1], [0.0, 0.4]]).view(bs,2).to(device)
    scale = torch.tensor([[4.4], [1.0]]).view(bs,1).to(device)
    rotation = torch.tensor([[-3.5], [0.3]]).view(bs,1).to(device)
    theta = torch.cat([scale, location, rotation], 1)

    preds = defaultdict(int)

    all_composites = []
    preds = defaultdict(int)
    trial_count = 5000
    for i in range(trial_count):
        out_pred, comb_img = visual_synthesis_helper(vae, num1, num2, theta, bs, shape_label, s_classes, object_classifier, folder_path, save_images=i==0)
        preds[out_pred] += 1
        all_composites.append(comb_img)

    # stack and compare
    all_imgs = torch.stack(all_composites)   # (500, 3, 64, 64)
    mean_img = all_imgs.mean(0)
    std_img  = all_imgs.std(0)
    print(f"Pixel std across trials: min={std_img.min():.6f} max={std_img.max():.6f} mean={std_img.mean():.6f}")

    save_image(torch.cat([all_imgs[:25].view(25,3,64,64), mean_img.unsqueeze(0), std_img.unsqueeze(0) / std_img.max()], 0),
               f'{folder_path}variation_across_trials.png', nrow=27, normalize=False, pad_value=0.6)


    accuracy = preds[object_names.index('umbrella')] / trial_count
    with open(f"{folder_path}results.txt", "w") as f:
        for cls in preds:
            f.write(f'{object_names[cls]}: {preds[cls]} out of {trial_count}\n')
        f.write(f'Accuracy: {accuracy:.4f}')
    
@torch.no_grad()
def fig_visual_synthesis_clock(vae: VAE_CNN, shape_label, s_classes, object_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    vae.eval()
    bs = 2
    num1 = 0 #24
    num2 = 31

    location = torch.tensor([[-0.2, 0.0], [0.05, 0.0]]).view(bs,2).to(device)
    scale = torch.tensor([[3.4], [0.85]]).view(bs,1).to(device)
    rotation = torch.tensor([[0], [0.4]]).view(bs,1).to(device)
    theta = torch.cat([scale, location, rotation], 1)

    preds = defaultdict(int)
    trial_count = 5000
    
    all_composites = []
    preds = defaultdict(int)
    trial_count = 500
    for i in range(trial_count):
        out_pred, comb_img = visual_synthesis_helper(vae, num1, num2, theta, bs, shape_label, s_classes, object_classifier, folder_path, save_images=i==0)
        preds[out_pred] += 1
        all_composites.append(comb_img)

    # stack and compare
    all_imgs = torch.stack(all_composites)   # (500, 3, 64, 64)
    mean_img = all_imgs.mean(0)
    std_img  = all_imgs.std(0)
    print(f"Pixel std across trials: min={std_img.min():.6f} max={std_img.max():.6f} mean={std_img.mean():.6f}")

    save_image(torch.cat([all_imgs[:25].view(25,3,64,64), mean_img.unsqueeze(0), std_img.unsqueeze(0) / std_img.max()], 0),
               f'{folder_path}variation_across_trials.png', nrow=27, normalize=False, pad_value=0.6)




    accuracy = preds[object_names.index('clock')] / trial_count
    with open(f"{folder_path}results.txt", "w") as f:
        for cls in preds:
            f.write(f'{object_names[cls]}: {preds[cls]} out of {trial_count}\n')
        f.write(f'Accuracy: {accuracy:.4f}')

@torch.no_grad()
def fig_visual_synthesis_boat(vae: VAE_CNN, shape_label, s_classes, object_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    vae.eval()
    bs = 2
    num1 = 13
    num2 = 25

    location = torch.tensor([[0.0, 0.0], [0.05, -0.3]]).view(bs,2).to(device)
    scale = torch.tensor([[3.1], [0.5]]).view(bs,1).to(device)
    rotation = torch.tensor([[4.0], [0.0]]).view(bs,1).to(device)
    theta = torch.cat([scale, location, rotation], 1)

    preds = defaultdict(int)
 
    all_composites = []
    preds = defaultdict(int)
    trial_count = 5000
    for i in range(trial_count):
        out_pred, comb_img = visual_synthesis_helper(vae, num1, num2, theta, bs, shape_label, s_classes, object_classifier, folder_path, save_images=i==0)
        preds[out_pred] += 1
        all_composites.append(comb_img)

    # stack and compare
    all_imgs = torch.stack(all_composites)   # (500, 3, 64, 64)
    mean_img = all_imgs.mean(0)
    std_img  = all_imgs.std(0)
    print(f"Pixel std across trials: min={std_img.min():.6f} max={std_img.max():.6f} mean={std_img.mean():.6f}")

    save_image(torch.cat([all_imgs[:25].view(25,3,64,64), mean_img.unsqueeze(0), std_img.unsqueeze(0) / std_img.max()], 0),
               f'{folder_path}variation_across_trials.png', nrow=27, normalize=False, pad_value=0.6)




    accuracy = preds[object_names.index('sailboat')] / trial_count
    with open(f"{folder_path}results.txt", "w") as f:
        for cls in preds:
            f.write(f'{object_names[cls]}: {preds[cls]} out of {trial_count}\n')
        f.write(f'Accuracy: {accuracy:.4f}')

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

def rgb_to_lab_norm(imgs_rgb: torch.Tensor,
                    l_min: float = 50,
                    l_max: float = 70) -> torch.Tensor:
    """
    Convert (B, 3, H, W) RGB in [0,1] to normalized Lab in [0,1].
    L* normalized using the same [l_min, l_max] range as the dataloader.
    a*, b* normalized over [-128, 127] (full Lab gamut).
    returns: (B, 3, H, W)
    """
    linear = torch.where(
        imgs_rgb <= 0.04045,
        imgs_rgb / 12.92,
        ((imgs_rgb + 0.055) / 1.055) ** 2.4,
    )

    X = (0.4124564 * linear[:, 0] +
         0.3575761 * linear[:, 1] +
         0.1804375 * linear[:, 2]).unsqueeze(1)
    Y = (0.2126729 * linear[:, 0] +
         0.7151522 * linear[:, 1] +
         0.0721750 * linear[:, 2]).unsqueeze(1)
    Z = (0.0193339 * linear[:, 0] +
         0.1191920 * linear[:, 1] +
         0.9503041 * linear[:, 2]).unsqueeze(1)

    X = X / 0.95047
    Z = Z / 1.08883

    delta = 6.0 / 29.0
    def f(t):
        return torch.where(t > delta ** 3,
                           t.pow(1.0 / 3.0),
                           t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(X), f(Y), f(Z)

    L     = 116.0 * fy - 16.0          # [0, 100]
    a_star = 500.0 * (fx - fy)         # [-128, 127] approx
    b_star = 200.0 * (fy - fz)

    # Normalize L* using dataloader range — same mapping as Colorize_specific
    L_norm = ((L - l_min) / (l_max - l_min)).clamp(0.0, 1.0)
    a_norm = ((a_star + 128.0) / 255.0).clamp(0.0, 1.0)
    b_norm = ((b_star + 128.0) / 255.0).clamp(0.0, 1.0)

    return torch.cat([L_norm, a_norm, b_norm], dim=1)   # (B, 3, H, W)

def visualize_color_objective(imgs, imgsize,
                               l_min: float = 50,
                               l_max: float = 70):
    """
    imgs : (B, 3, H, W) in [0,1]
    returns : (B, 3, H, W) in [0,1] — flat-color patch using mean Lab
    """
    B = imgs.shape[0]
    imgs = imgs.view(B, 3, imgsize, imgsize)

    brightness = imgs.max(dim=1, keepdim=True).values
    mask = (brightness > 0.15).float()
    fg_count = mask.sum(dim=[2, 3], keepdim=True).clamp(min=1)

    lab = rgb_to_lab_norm(imgs, l_min, l_max)                          # (B, 3, H, W)
    mean_lab = (lab * mask).sum(dim=[2, 3], keepdim=True) / fg_count   # (B, 3, 1, 1)
    flat_lab = mean_lab.expand(B, 3, imgsize, imgsize)                 # (B, 3, H, W)

    # Denormalize back to raw Lab for conversion
    L      = flat_lab[:, 0:1] * (l_max - l_min) + l_min   # [l_min, l_max]
    a_star = flat_lab[:, 1:2] * 255.0 - 128.0
    b_star = flat_lab[:, 2:3] * 255.0 - 128.0

    # Lab -> XYZ
    fy = (L + 16.0) / 116.0
    fx = a_star / 500.0 + fy
    fz = fy - b_star / 200.0

    delta = 6.0 / 29.0
    def f_inv(t):
        return torch.where(t > delta,
                           t ** 3,
                           3 * delta ** 2 * (t - 4.0 / 29.0))

    X = f_inv(fx) * 0.95047
    Y = f_inv(fy)
    Z = f_inv(fz) * 1.08883

    M_inv = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=imgs.dtype, device=imgs.device)

    xyz = torch.cat([X, Y, Z], dim=1).view(B, 3, -1)
    rgb_linear = torch.einsum('cd,bdp->bcp', M_inv, xyz).view(B, 3, imgsize, imgsize).clamp(0, 1)

    rgb = torch.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        1.055 * rgb_linear.pow(1.0 / 3.0) - 0.055,
    )
    return rgb.clamp(0.0, 1.0)

@torch.no_grad()
def functionality_test(vae: VAE_CNN, shape_label, s_classes, color_label, c_classes, folder_path: str):
    vae.eval()
    #vae.skip_bn.train()
    device = next(vae.parameters()).device
    emnist_targetset = [2, 3, 12, 15, 25]
    mnist_transforms = {'retina':True, 'colorize':True, 'rotate':False, 'scale':True} #, 'target_set':emnist_targetset}
    emnist_loader= Dataset('square', mnist_transforms).get_loader(50)
    
    dataiter_emnist = iter(emnist_loader)
    data, labels = next(dataiter_emnist)
    data = data[1].cuda()
    sample_size = 25
    sample = data[:sample_size].to(device)   #the actual image


    # color label testing
    x = sample
    target = visualize_color_objective(x, imgsize, 50, 70)
    # END COLOR LABELS
    name = ''
    vae.dropout.p = 0
    #sample = data[:sample_size].to(device)   #the actual image
    with torch.no_grad():
        if 'quickdraw' in name:
            recon, _, _, _, _, _, _ = vae(sample, 'cropped_object', ['object', 'color'])
            shape, _, _, _, _, _, _ = vae(sample, 'object', ['object'])
        else:
            recon, _, _, _, _, _, _ = vae(sample, 'cropped', ['shape', 'color'])
            shape, _, _, _, _, _, _ = vae(sample, 'shape', ['shape'])
        color, _, _, _, _, _, _ = vae(sample, 'color', ['color'])
        skip, _, _, _, _, _, _ = vae(sample, 'skip_cropped', ['skip'])

    vae.dropout.p = 0.1 
     
    output_img = torch.cat([sample.view(sample_size, 3, imgsize, imgsize)[:25], recon.view(sample_size, 3, imgsize, imgsize)[:25], skip.view(sample_size, 3, imgsize, imgsize)[:25],
                       shape.view(sample_size, 3, imgsize, imgsize)[:25], color.view(sample_size, 3, imgsize, imgsize)[:25], target.view(sample_size, 3, imgsize, imgsize)[:25]], 0)
     
    rows = 6;    

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
    text_tensor = text_to_tensor("Image / both maps recon / skip recon / shape recon/color recon / color objective ",header_height,width)
    new_tensor[:, :header_height, :] = text_tensor
    save_image(new_tensor,f'{folder_path}cropped_sample_{name}.png',
            nrow=1, normalize=False)


@torch.no_grad()
def fig_generative_noise_1dPCA(vae: VAE_CNN, shape_label, s_classes, color_label, c_classes, folder_path: str, load_data: bool = False):
    vae.eval()
    device = next(vae.parameters()).device
    n = 10

    # base: green Z from label networks
    num_label    = F.one_hot(torch.tensor([35]).to(device), num_classes=s_classes).float()
    col_label    = F.one_hot(torch.tensor([1 ]).to(device), num_classes=c_classes).float()
    z_shape_base = shape_label(num_label, n=0)   # (1, shape_z_dim)
    z_color_base = color_label(col_label, n=0)   # (1, color_z_dim)

    # compute PCA axes from a large real batch — gives meaningful directions to walk
    loader = Dataset('emnist', {'retina': False, 'colorize': True}, train=True).get_loader(2000)
    imgs, _ = next(iter(loader))
    act     = vae.activations(imgs.to(device), False)
    z_shape_all = act['shape']
    z_color_all = act['color']

    _, _, Vs = torch.pca_lowrank(z_shape_all - z_shape_all.mean(0), q=1)
    _, _, Vc = torch.pca_lowrank(z_color_all - z_color_all.mean(0), q=1)
    pc_shape = Vs[:, 0]   # dominant shape direction
    pc_color = Vc[:, 0]   # dominant color direction

    # step range in units of std along each PC
    shape_std = ((z_shape_all - z_shape_all.mean(0)) @ pc_shape).std()
    color_std = ((z_color_all - z_color_all.mean(0)) @ pc_color).std()
    shape_steps = torch.linspace(0, shape_std * 3, n).to(device)
    color_steps = torch.linspace(0, color_std * 3, n).to(device)

    # top row: walk along PC1 of shape, color fixed at green Z
    shape_recons = []
    for s in shape_steps:
        z_s = z_shape_base + s * pc_shape.unsqueeze(0)
        shape_recons.append(vae.decoder_cropped(z_s, z_color_base))

    # left col: walk along PC1 of color, shape fixed at Z
    color_recons = []
    for s in color_steps:
        z_c = z_color_base + s * pc_color.unsqueeze(0)
        color_recons.append(vae.decoder_cropped(z_shape_base, z_c))

    # diagonal: both walk together
    joint_recons = [vae.decoder_cropped(z_shape_base, z_color_base)]  # clean base
    for s_s, s_c in zip(shape_steps, color_steps):
        z_s = z_shape_base + s_s * pc_shape.unsqueeze(0)
        z_c = z_color_base + s_c * pc_color.unsqueeze(0)
        joint_recons.append(vae.decoder_cropped(z_s, z_c))

    recon_grid = build_gen_grid(joint_recons, shape_recons, color_recons, n)
    save_image(recon_grid, f'{folder_path}noisewalk.png', pad_value=0.6)


@torch.no_grad()
def fig_generative_noise_2DPCA(vae: VAE_CNN, shape_label, s_classes, color_label, c_classes, folder_path: str, load_data: bool = False):
    vae.eval()
    device = next(vae.parameters()).device
    n = 10
    step_size = 1   # fraction of std per step — tune this

    # base: green Z from label networks
    num_label    = F.one_hot(torch.tensor([35]).to(device), num_classes=s_classes).float()
    col_label    = F.one_hot(torch.tensor([1 ]).to(device), num_classes=c_classes).float()
    z_shape_base = shape_label(num_label, n=0)
    z_color_base = color_label(col_label, n=0)

    # compute 2D PCA manifolds from real encoded images
    loader = Dataset('emnist', {'retina': False, 'colorize': True}, train=True).get_loader(2000)
    imgs, _ = next(iter(loader))
    act = vae.activations(imgs.to(device), False)
    z_shape_all = act['shape']
    z_color_all = act['color']

    shape_mean = z_shape_all.mean(0)
    color_mean = z_color_all.mean(0)

    _, _, Vs = torch.pca_lowrank(z_shape_all - shape_mean, q=2)
    _, _, Vc = torch.pca_lowrank(z_color_all - color_mean, q=2)
    spc1, spc2 = Vs[:, 0], Vs[:, 1]   # shape manifold axes
    cpc1, cpc2 = Vc[:, 0], Vc[:, 1]   # color manifold axes

    # std along each PC — sets the scale of a meaningful step
    s_std1 = ((z_shape_all - shape_mean) @ spc1).std().item()
    s_std2 = ((z_shape_all - shape_mean) @ spc2).std().item()
    c_std1 = ((z_color_all - color_mean) @ cpc1).std().item()
    c_std2 = ((z_color_all - color_mean) @ cpc2).std().item()

    # random walk in 2D PCA space: each step is a random displacement
    # along PC1 and PC2, scaled by the real data std
    def pca_walk(base, mean, pc1, pc2, std1, std2):
        centered = base - mean
        s1 = (centered @ pc1).item()
        s2 = (centered @ pc2).item()
        residual = centered - s1 * pc1 - s2 * pc2

        # pick one random direction in the 2D plane and march along it
        direction = torch.randn(2).to(device)
        direction = direction / direction.norm()   # unit vector in 2D PCA space

        zs = []
        for i in range(n):
            ds1 = direction[0].item() * step_size * std1 * i
            ds2 = direction[1].item() * step_size * std2 * i
            zs.append((mean + (s1 + ds1) * pc1 + (s2 + ds2) * pc2 + residual).unsqueeze(0))
        return zs

    shape_walk = pca_walk(z_shape_base, shape_mean, spc1, spc2, s_std1, s_std2)
    color_walk = pca_walk(z_color_base, color_mean, cpc1, cpc2, c_std1, c_std2)

    # decode each trajectory
    print(z_color_base)
    shape_recons = [vae.decoder_cropped(z, z_color_base) for z in shape_walk]
    color_recons = [vae.decoder_cropped(z_shape_base, z) for z in color_walk]

    joint_recons = [vae.decoder_cropped(z_shape_base, z_color_base)]  # clean start
    for zs, zc in zip(shape_walk, color_walk):
        joint_recons.append(vae.decoder_cropped(zs, zc))

    recon_grid = build_gen_grid(joint_recons, shape_recons, color_recons, n)
    save_image(recon_grid, f'{folder_path}noisewalk.png', pad_value=0.6)

@torch.no_grad()
def fig_binding_addressability(vae: VAE_CNN, color_classifier, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()
    if load_data is False:
        print("addressability figure")
        # store 2 digits, generate activations of greyscaled rep of 1 of the digits, retrieve from BP using that shape as a cue
        
        numimg = 2
        dataset = Dataset('emnist',{'retina':False, 'colorize':True, 'rotate':False, 'scale':True}, train=False)  #two random items
        dataset_2 = Dataset('emnist',{'retina':False, 'colorize':True, 'rotate':False, 'scale':True, 'target_set':[10]}, train=False)  # always store/retrieve just the same type of letter

        #now run a batch of trials for each of the two conditions:
        #compute how often the correct token & color is retrieved  
        #return the last trial as a visual example for the figure 
        print('generating two random items')
        fig_data_list_r, accuracy_r = store_retrieve2('emnist_rand', dataset, vae, color_classifier, numimg, folder_path)  #two random ones
        print('generating two As')
        fig_data_list_2,accuracy_2 = store_retrieve2('emnist_2', dataset_2, vae, color_classifier, numimg, folder_path)   #always the same letter 
    
        data_to_pickle = {
            "fig_data_list_r": fig_data_list_r,
            "fig_data_list_2": fig_data_list_2,
            "accuracy_r": accuracy_r,
            "accuracy_2": accuracy_2
        }

        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        
        # load plotting data
        loaded_data = joblib.load(pkl_path)
        fig_data_list_r = loaded_data["fig_data_list_r"]
        fig_data_list_2 = loaded_data["fig_data_list_2"]
        accuracy_r = loaded_data["accuracy_r"]
        accuracy_2 = loaded_data["accuracy_2"]

    save_image(torch.cat(fig_data_list_r, 0), f'{folder_path}mnist_rand-addressability.png',
                nrow=numimg, normalize=False, pad_value=0.6)

    save_image(torch.cat(fig_data_list_2, 0), f'{folder_path}mnist_2-addressability.png',
            nrow=numimg, normalize=False, pad_value=0.6)
        # --- bar graph of addressability accuracy ---
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ['Correct Token', 'Correct Color']
    random_vals = [accuracy_r['correct_token'], accuracy_r['correct_color']]
    same_vals = [accuracy_2['correct_token'], accuracy_2['correct_color']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width/2, random_vals, width, label='Random', color='#4878CF')
    bars2 = ax.bar(x + width/2, same_vals, width, label='Same', color='#D65F5F')

    ax.set_ylabel('Proportion Correct')
    ax.set_title('Binding Pool Addressability')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{folder_path}addressability_bargraph.png', dpi=300)
    plt.close()
    print(f"Bar graph saved to {folder_path}addressability_bargraph.png")



def store_retrieve2(trial_name: str, dataset, vae: VAE_CNN, color_classifier, numimg, folder_path: str):
    #stores two items in memory with different colors and then retrieves based on the shape 


    bpPortion = int(token_overlap *bpsize)

    test_loader = cycle(dataset.get_loader(numimg))
    dataiter = iter(test_loader)
    total_trials = 1000
    out_predictions = 0
    token_predictions = 0
    
    green = Colorize_specific(1)
    red = Colorize_specific(0)

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
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg, normalize_factor)
        
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

        _, maxtoken =torch.max(tokenactivation,0) #which token has the most activation
        BPOut_cued[0, notLink_all[maxtoken, :]] = 0

        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg, normalize_factor)
        BP_act_out_cued = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut_cued, Tokenbindings, BP_activations_sc, numimg, normalize_factor)
        
        #cued_prediction = shape_classifier.predict(BP_act_out_cued['shape'][0].view(1,-1).cpu())

        cued_color_prediction = color_classifier.predict(BP_act_out_cued['color'][maxtoken].view(1,-1).cpu())

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
    accuracy = {'correct_token': token_predictions / total_trials, 
                'correct_color': out_predictions / total_trials,
                'total_trials': total_trials}

    
    # save params used in this simulation run
    with open(f"{folder_path}params.txt", "w") as f:
        f.write(f"bpsize: {bpsize}\n")
        f.write(f"token_overlap: {token_overlap}\n")
        f.write(f"bpPortion: {bpPortion}\n")
    
    return fig_data_list, accuracy



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

@torch.no_grad()
def fig_efficient_rep(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    #used in the interference panel
        #demonstrates that the same stimulus when reconstructed through L1 accumulates interference more quickly..
            #except that the metrics don't support this, even though its visually quite bad
            #this is labeled as novel but it's not novel, we just L1 for the familiar stimuli

    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    vae.eval()

    if load_data is False:
        print('generating Figure efficient reconstruction plot')
        retina_size = 100
        imgsize = 28
        #bpsize = 10000         #size of the binding pool
        #token_overlap = 0.3
        bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
        numimg = 7
        ss1 = 1
        ss4 = 4
        #make the data loader
        test_loader_mnist = Dataset('mnist',{'colorize':True}, train=True).get_loader(numimg)
        test_loader_emnist = Dataset('emnist',{'retina':False, 'colorize':True, 'rotate':False, 'scale':True, 'target_set':[10, 11, 12, 13, 14, 15]}, train=True).get_loader(numimg)

        dataiter_mnist = iter(test_loader_mnist)
        dataiter_emnist = iter(test_loader_emnist)
        fam_1 = []
        nov_1 = []
        fam_4 = []
        nov_4 = []
        trialcount = 10
        for count in range(0,trialcount):
            data_mnist, labels = next(dataiter_mnist)
            data_emnist, labels = next(dataiter_emnist)
            #data_emnist = imgs # Bengali chars not emnist
            
            mnist_sample = data_mnist[:ss4].cuda()
            emnist_sample = data_emnist[:ss4].cuda()
            
            #push the images through the model
            mnist_act = vae.activations(emnist_sample.view(-1,3,28,28), False)
            emnist_act = vae.activations(emnist_sample.view(-1,3,28,28), False)
            
            emnist_shape_act = emnist_act['shape']
            emnist_color_act = emnist_act['color']

            mnist_shape_act = mnist_act['shape']
            mnist_color_act = mnist_act['color']
            mnist_l1_act = mnist_act['skip']


            BP_activations_fam_1 = {'shape': [emnist_shape_act[:ss1].view(ss1,-1), 1], 'color': [emnist_color_act[:ss1].view(ss1,-1), 1]} # 2 familiar in shape/color
            BP_activations_nov_1_sc = {'shape': [mnist_shape_act[:ss1].view(ss1,-1), 1], 'color': [mnist_color_act[:ss1].view(ss1,-1), 1]} # 2 familiar in shape/color
            BP_activations_nov_1 = {'l1': [mnist_l1_act[:ss1].view(ss1,-1), 1]} # 2 novel in L1

            BP_activations_fam_4 = {'shape': [emnist_shape_act.view(ss4,-1), 1], 'color': [emnist_color_act.view(ss4,-1), 1]} # 4 familiar in shape/color
            BP_activations_nov_4_sc = {'shape': [mnist_shape_act.view(ss4,-1), 1], 'color': [mnist_color_act.view(ss4,-1), 1]} # 4 familiar in shape/color
            BP_activations_nov_4 = {'l1': [mnist_l1_act.view(ss4,-1), 1]} # 4 novel in L1

            # store and retrieve 1 familiar s/c maps
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_fam_1, ss1,normalize_factor)
            BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_fam_1, ss1,normalize_factor)
            shape_out_1, color_out_1 = BP_activations_out['shape'], BP_activations_out['color']

            # store and retrieve 1 familiar l1 act
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_nov_1, ss1,normalize_factor)
            BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_nov_1, ss1,normalize_factor)
            l1_out_1 = BP_activations_out['l1']
            
            
            # store and retrieve 4 familiar s/c maps
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_fam_4, ss4,normalize_factor)
            BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_fam_4, ss4,normalize_factor)
            shape_out_4, color_out_4 = BP_activations_out['shape'], BP_activations_out['color']

            # store and retrieve 4 familiar l1 act
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_nov_4, ss4,normalize_factor)
            BP_activations_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_nov_4, ss4,normalize_factor)
            l1_out_4 = BP_activations_out['l1']
    
            recon_fam_1 = vae.decoder_cropped(shape_out_1, color_out_1,0,0).cuda() #rgb_to_gray(vae.decoder_shape(shape_out_1, 0))#
            #recon_nov_1 = vae.decoder_cropped(shape_out_novel_1, color_out_novel_1,0,0).cuda() #rgb_to_gray(vae.decoder_shape(shape_out_1, 0))#
            recon_nov_1 = vae.decoder_skip_cropped(0, 0, 0, l1_out_1).cuda()
            recon_nov_1 = Ft.gaussian_blur(recon_nov_1, kernel_size=[3, 3], sigma=[1.0, 1.0])
            #recon_nov_1  = torch.where(recon_nov_1  < 0.15, torch.zeros_like(recon_nov_1 ),recon_nov_1 )
            #recon_nov_1 = torch.from_numpy(ndimage.median_filter(recon_nov_1.cpu().numpy(), size=(1,1,3,3))).to(recon_nov_1.device)

            recon_fam_4 = vae.decoder_cropped(shape_out_4, color_out_4,0,0).cuda() #rgb_to_gray(vae.decoder_shape(shape_out_4, 0))#
            #recon_nov_4 = vae.decoder_cropped(shape_out_novel_4, color_out_novel_4,0,0).cuda() #rgb_to_gray(vae.decoder_shape(shape_out_4, 0))#
            recon_nov_4 = vae.decoder_skip_cropped(0, 0, 0, l1_out_4).cuda()
        
            recon_nov_4 = Ft.gaussian_blur(recon_nov_4, kernel_size=[3, 3], sigma=[1.0, 1.0])
            #recon_nov_4  = torch.where(recon_nov_4  < 0.4, torch.zeros_like(recon_nov_4 ),recon_nov_4 )

            corr_fam_1 = batch_ssim(emnist_sample[:ss1], recon_fam_1)
            corr_nov_1 = batch_ssim(emnist_sample[:ss1], recon_nov_1)
            corr_fam_4 = batch_ssim(emnist_sample, recon_fam_4)
            corr_nov_4 = batch_ssim(emnist_sample, recon_nov_4)
            #corr_fam_1 = compute_correlation(emnist_sample[:ss1], recon_fam_1).item()
            #corr_nov_1 = compute_correlation(emnist_sample[:ss1], recon_nov_1).item()
            #corr_fam_4 = compute_correlation(emnist_sample, recon_fam_4).item()
            #corr_nov_4 = compute_correlation(emnist_sample, recon_nov_4).item()

            fam_1 += [corr_fam_1]
            nov_1 += [corr_nov_1]

            fam_4 += [corr_fam_4]
            nov_4 += [corr_nov_4]

        corr_fam_1 = sum(fam_1)/len(fam_1)
        corr_nov_1 = sum(nov_1)/len(nov_1)

        corr_fam_4 = sum(fam_4)/len(fam_4)
        corr_nov_4 = sum(nov_4)/len(nov_4)

        print(corr_nov_1, corr_nov_4)
        print(corr_fam_1, corr_fam_4)

        e = torch.zeros((1,3,28,28)).cuda()
        fig_data = [mnist_sample, torch.cat([recon_nov_1, e, e, e], 0), recon_nov_4, emnist_sample,
                   torch.cat([recon_fam_1, e, e, e], 0), recon_fam_4,]

        data_to_pickle = {
            "fig_data": fig_data,
            "ss1": ss1,
            "ss4": ss4,
            "corr_nov_1": corr_nov_1,
            "corr_nov_4": corr_nov_4,
            "corr_fam_1": corr_fam_1,
            "corr_fam_4": corr_fam_4,
        }

        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        
        # load plotting data
        loaded_data = joblib.load(pkl_path)
        ss1 = loaded_data["ss1"]
        ss4 = loaded_data["ss4"]
        corr_nov_1 = loaded_data["corr_nov_1"]
        corr_nov_4 = loaded_data["corr_nov_4"]
        corr_fam_1 = loaded_data["corr_fam_1"]
        corr_fam_4 = loaded_data["corr_fam_4"]

    save_image(
        torch.cat(fig_data, 0),
        f'{folder_path}efficient_recon_sample_ss1_ss4.png', pad_value=0.6,
        nrow=ss4, normalize=False)

    plt.figure()

#    sns.lineplot(
#        x=[ss1, ss4],
#        y=[corr_nov_1, corr_nov_4],
#        label='novel images (L1)'
#    )

    sns.lineplot(
        x=[ss1, ss4],
        y=[corr_fam_1, corr_fam_4],
        label='familiar images (feature maps)'
    )

    plt.xlabel('set size')
    plt.ylabel('SSIM')
    plt.title('Set Size vs. Reconstruction Quality')
    plt.legend()

    plt.savefig(f'{folder_path}efficient_recon.png')
    plt.close()


def feature_swap_trial(dataset, vae: VAE_CNN, numimg: int, imgsize: int):
    #used in the interference panel.  Loads up memories with squares at different locations

    cx, cy, r = 14, 14, 5  # center of 28x28, radius 5

    test_loader = cycle(dataset.get_loader(numimg))
    dataiter = iter(test_loader)
    #bpsize = 10000        #size of the binding pool
    #token_overlap =0.3
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

    target_dists = []
    nontarget_dists = []
    token_swap = 0
    trial_count =200
    for _ in range(trial_count):
        crop_imgs = next(dataiter)[0].cuda()  #grab some squares

        imgs = torch.zeros(numimg,3,64,64).cuda()
        locations = sample_points(64 - imgsize, 64 - imgsize, k=numimg, min_dist=10)
        colors = random.sample(range(10), numimg)
        for i in range(numimg):
            x, y = locations[i]
            colorizer = Colorize_specific(colors[i])
            colored_crop = convert_tensor(colorizer(convert_image(crop_imgs[i].cpu()))).cuda()
            imgs[i, :, x:x+imgsize, y:y+imgsize] = colored_crop
        
        #push the images through the encoder one at a time
        activations = vae.activations(imgs.view(-1,3,64, 64), True)
        shape_act = activations['shape']
        color_act = activations['color']
        location_act = activations['location']

        cue_activations = vae.activations(imgs[0].view(-1,3,64, 64), True)
        cue_location_act = cue_activations['location'] 
        
        BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1],
                             'color': [color_act.view(numimg,-1), 1],
                             'location': [location_act.view(numimg,-1), 1]}
        
        #now store items
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg, normalize_factor)
        
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

        _, maxtoken = torch.max(tokenactivation, 0)
        BPOut_cued[0, notLink_all[maxtoken, :]] = 0

        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg, normalize_factor)
        BP_act_out_cued = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut_cued, Tokenbindings, BP_activations_sc, numimg, normalize_factor)

        # decode retrieved color
        recon_cued = vae.decoder_color(0, BP_act_out_cued['color'][maxtoken].view(1, -1))
        cued_patch = recon_cued[0, :, cy-r:cy+r, cx-r:cx+r].mean(dim=[1, 2])

        # decode target's original color
        recon_target = vae.decoder_color(0, color_act[0].view(1, -1))
        target_patch = recon_target[0, :, cy-r:cy+r, cx-r:cx+r].mean(dim=[1, 2])

        d_target = torch.norm(cued_patch - target_patch).item()
        target_dists.append(d_target)

        if numimg > 1:
            d_others = []
            for i in range(1, numimg):
                recon_other = vae.decoder_color(0, color_act[i].view(1, -1))
                other_patch = recon_other[0, :, cy-r:cy+r, cx-r:cx+r].mean(dim=[1, 2])
                d_others.append(torch.norm(cued_patch - other_patch).item())
            nontarget_dists.append(npy.mean(d_others))


        # same for non-targets
        if numimg > 1:
            d_others = []
            for i in range(1, numimg):
                recon_other = vae.decoder_color(0, color_act[i].view(1, -1))
                other_patch = recon_other[0, :, cy-r:cy+r, cx-r:cx+r].mean(dim=[1, 2])
                d_others.append(torch.norm(cued_patch - other_patch).item())
            nontarget_dists.append(npy.mean(d_others))


        if maxtoken != 0:
            token_swap += 1


    # example images from last trial
    orig_recon = vae.decoder_cropped(shape_act, color_act, 0, 0)
    orig_recon = F.interpolate(orig_recon, size=(64, 64), mode='bilinear', align_corners=False)

    grey_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=imgs.device).view(1, 3, 1, 1)
    grey_cue = (imgs[0:1] * grey_weights).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    pad_count = numimg - 1 if numimg > 1 else 0
    pad = torch.zeros(pad_count, 3, 64, 64).cuda()
    cue_row = torch.cat([grey_cue, pad], 0) if numimg > 1 else grey_cue

    recon_cued_img = vae.decoder_cropped(
        BP_act_out_cued['shape'][maxtoken].view(1, -1),
        BP_act_out_cued['color'][maxtoken].view(1, -1), 0, 0)
    recon_cued_img = F.interpolate(recon_cued_img, size=(64, 64), mode='bilinear', align_corners=False)
    cued_row = torch.cat([recon_cued_img, pad], 0) if numimg > 1 else recon_cued_img

    example_imgs = torch.cat([imgs, orig_recon, cue_row, cued_row], 0)
    return {
        'token_swap': token_swap / trial_count,
        'target_dist': npy.mean(target_dists),
        'nontarget_dist': npy.mean(nontarget_dists) if nontarget_dists else 0.0,
        'example_imgs': example_imgs,
        'numimg': numimg
    }

    

@torch.no_grad()
def fig_feature_swap(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'

    if load_data is False:
        vae.eval()
#        dataset = Dataset('emnist', {'retina': False, 'colorize': False, 'rotate': False, 'scale': True}, train=False)
        dataset = Dataset('square', {'retina': False, 'colorize': False, 'rotate': False, 'scale': True}, train=False)

        token_swap_rates = []
        target_dists = []
        nontarget_dists = []

        for numimg in range(1, 8):
            result = feature_swap_trial(dataset, vae, numimg, imgsize=imgsize)
            token_swap_rates.append(result['token_swap'])
            target_dists.append(result['target_dist'])
            nontarget_dists.append(result['nontarget_dist'])
            save_image(result['example_imgs'],
            f'{folder_path}feature_swap_example_{numimg}.png',
            nrow=result['numimg'], normalize=False, pad_value=0.6)

        data_to_pickle = {
            'token_swap_rates': token_swap_rates,
            'target_dists': target_dists,
            'nontarget_dists': nontarget_dists
        }
        joblib.dump(data_to_pickle, pkl_path)

    else:
        if not os.path.exists(pkl_path):
            raise Exception(f"No data exists for plot: {folder_path}{log_function_name()}")
        loaded = joblib.load(pkl_path)
        token_swap_rates = loaded['token_swap_rates']
        target_dists = loaded['target_dists']
        nontarget_dists = loaded['nontarget_dists']

    # token swap rate
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, 8), token_swap_rates, 'o-')
    ax.set_xlabel('Set size')
    ax.set_ylabel('Token swap rate')
    ax.set_title('Token Swap Rate by Set Size')
    plt.tight_layout()
    plt.savefig(f'{folder_path}token_swap_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    # color distance: target vs non-target
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, 8), target_dists, 'o-', label='Target')
    ax.plot(range(2, 8), nontarget_dists[1:], 's-', label='Non-target mean')
    ax.set_xlabel('Set size')
    ax.set_ylabel('L2 distance')
    ax.set_title('Retrieved Color Distance: Target vs Non-Target')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{folder_path}feature_swap_distance.png', dpi=300, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def fig_encoding_flexibility(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    pkl_path = f'{folder_path}{log_function_name()}-figure_data.pkl'
    
    vae.eval()
    print("encoding flexibility figure")
    numimg = 1

    #bpsize = 10000#00         #size of the binding pool
    #token_overlap =0.35
    bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
    targetset = list(range(0, 4))
    targetset.append(15)
    #dataset = Dataset('emnist',{'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'target_set':targetset}, train=False)
    dataset = Dataset('emnist',{'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'target_set':[10, 12, 15, 18, 20], 'location_targets': {(-1, -1): [10, 12, 15, 18, 20]}}, train=True)
    test_loader = dataset.get_loader(numimg)
    dataiter = iter(test_loader)
    imgs = next(dataiter)[0][0].cuda()

    #push the images through the encoder
    activations = vae.activations(imgs.view(-1,3,64,64), True)
    shape_act = activations['shape']
    color_act = activations['color']
    location_act = activations['location']
    scale_act = activations['scale']
    theta_orig = torch.cat([scale_act, location_act], 1)
    color_degraded = []
    shape_degraded = []
    crop, theta = vae.stn_encode(imgs.view(-1,3,64,64))

    # degrade shape encoding weight: 1 -> 0.2
    for n in range (1,10,2):
        BP_activations_sc = {'shape': [shape_act.view(numimg,-1), 1/n], 'color': [color_act.view(numimg,-1), 1], 
                             'location': [location_act.view(numimg,-1), 1], 'scale': [scale_act.view(numimg,-1), 1]}
        
        #now store/retrieve from L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg,normalize_factor)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg,normalize_factor)

        # then through BP
        #bp_crop_recon = vae.decoder_cropped(BP_act_out['shape'], BP_act_out['color'])

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
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg,normalize_factor)
        BP_act_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg,normalize_factor)
        
        shape_out_BP, color_out_BP = BP_act_out['shape'], BP_act_out['color']
        location_out_BP, scale_out_BP = BP_act_out['location'], BP_act_out['scale']

        theta = torch.cat([scale_out_BP, location_out_BP], 1)
        BP_retinal_recon = vae.decoder_retinal(shape_out_BP, color_out_BP, theta)
        color_degraded += [BP_retinal_recon]
        location_degraded = []
    # degrade location encoding weight: 1 -> 0.2
        
    for n in range(1, 10, 2):
        BP_activations_sc = {'shape':    [shape_act.view(numimg,-1),    1],
                            'color':    [color_act.view(numimg,-1),    1],
                            'location': [location_act.view(numimg,-1), 1/n],
                            'scale':    [scale_act.view(numimg,-1),    1]}
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, BP_activations_sc, numimg, normalize_factor)
        BP_act_out = BPTokens_retrieveByToken(bpsize, bpPortion, BPOut, Tokenbindings, BP_activations_sc, numimg, normalize_factor)
        theta = torch.cat([BP_act_out['scale'], BP_act_out['location']], 1)
        location_degraded += [vae.decoder_retinal(BP_act_out['shape'], BP_act_out['color'], theta)]

        degraded = [x for triple in zip(shape_degraded, color_degraded, location_degraded) for x in triple]
        degraded = torch.cat(degraded, 0)
        sample = imgs[0: numimg].view(numimg, 3, 64, 64)
        save_image(torch.cat([sample, sample, sample, degraded], 0), f'{folder_path}figure2b.png',
                nrow=numimg*3, normalize=False, pad_value=0.6)
    
    with open(f"{folder_path}params.txt", "w") as f:
        f.write(f"bpsize: {bpsize}\n")
        f.write(f"token_overlap: {token_overlap}\n")
        f.write(f"bpPortion: {bpPortion}\n")


@torch.no_grad()
def fig_allstims(vae: VAE_CNN, folder_path: str, load_data: bool = False):
    from PIL import Image as PILImg, ImageDraw as PILDraw

    vae.eval()
    device = next(vae.parameters()).device

    N_EX   = 3
    N_SHP  = 36
    QD_CLS = [0, 2, 8, 10, 11]
    N_TOT  = N_SHP + len(QD_CLS)   # 41

    col_labels = (
        ['0','1','2','3','4','5','6','7','8','9'] +
        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
        [object_names[i][:4] for i in QD_CLS]
    )

    def fill_bins(loader, bins, needed_keys, n_ex):
        for imgs, labels in loader:
            for i in range(imgs.size(0)):
                lbl = labels[0][i].item()
                if lbl in bins and len(bins[lbl]) < n_ex:
                    bins[lbl].append(imgs[i].cpu())
            if all(len(bins[k]) >= n_ex for k in needed_keys):
                break

    shape_bins = {i: [] for i in range(N_SHP)}
    fill_bins(Dataset('mnist',  {'retina': False, 'colorize': True}, train=True).get_loader(256),
              shape_bins, range(10), N_EX)
    fill_bins(Dataset('emnist', {'retina': False, 'colorize': True}, train=True).get_loader(256),
              shape_bins, range(10, 36), N_EX)

    obj_bins = {cls: [] for cls in QD_CLS}
    fill_bins(Dataset('quickdraw', {'retina': False, 'colorize': True, 'target_set': QD_CLS},
                      train=True).get_loader(256),
              obj_bins, QD_CLS, N_EX)

    shape_recons = {}
    for lbl in range(N_SHP):
        batch = torch.stack(shape_bins[lbl][:N_EX]).to(device)
        shape_recons[lbl] = vae.decoder_shape(vae.activations(batch, False)['shape'])

    obj_recons = {}
    for cls in QD_CLS:
        batch = torch.stack(obj_bins[cls][:N_EX]).to(device)
        obj_recons[cls] = vae.decoder_object(vae.activations(batch, False)['object'])

    cells = []
    for row in range(N_EX):
        for lbl in range(N_SHP):
            cells.append(shape_recons[lbl][row])
        for cls in QD_CLS:
            cells.append(obj_recons[cls][row])

    W, H = N_TOT * imgsize, N_EX * imgsize
    grid_hw = (torch.stack(cells)
               .view(N_EX, N_TOT, 3, imgsize, imgsize)
               .permute(2, 0, 3, 1, 4).contiguous()
               .view(3, H, W))

    sep_x = N_SHP * imgsize
    grid_hw[:, :, sep_x] = 0.25

    hdr_pil = PILImg.new('RGB', (W, 16), (220, 220, 220))
    draw = PILDraw.Draw(hdr_pil)
    for c, name in enumerate(col_labels):
        draw.text((c * imgsize + 2, 2), name, fill=(0, 0, 0))
    #hdr = transforms.ToTensor()(hdr_pil)
    hdr = transforms.ToTensor()(hdr_pil).to(device)
    hdr[:, :, sep_x] = 0.25

    full = torch.cat([hdr, grid_hw], dim=1)
    save_image(full.unsqueeze(0), f'{folder_path}fig_identity_recon_grid.png', nrow=1, normalize=False)

        