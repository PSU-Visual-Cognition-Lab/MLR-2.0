#MNIST VAE retreived from https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

# Modifications:
#Colorize transform that changes the colors of a grayscale image
#colors are chosen from 10 options:
colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]
#specified in "colorvals" variable below
#also there is a skip connection from the first layer to the last layer to enable reconstructions of new stimuli
#and the VAE bottleneck is split, having two different maps
#one is trained with a loss function for color only (eliminating all shape info, reserving only the brightest color)
#the other is trained with a loss function for shape only
#print('pause for 1.5hr while previous run finishes')
#import time
#time.sleep(1.5*(60**2)) # wait 1 hr before starting

# prerequisites
import torch
import sys
import os
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
from scipy import stats
import gc
from PIL import Image

#internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MLR_src.dataset_builder import Dataset
from MLR_src.mVAE import *
from simulation_src.tokens_capacity import *
from MLR_src.BP_functions import BPTokens_binding_all, BPTokens_retrieveByToken, BPTokens_storage, BPTokens_with_labels


from PIL import Image, ImageOps, ImageEnhance#, __version__ as PILLOW_VERSION
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA')
else:
    device = 'cpu'

modelNumber= 1 #which model should be run, this can be 1 through 10

#folder_path = f'output{modelNumber}' # the output folder for the trained model versions

#if not os.path.exists(folder_path):
 #   os.mkdir(folder_path)

#load_checkpoint('output/checkpoint_threeloss_singlegrad200_smfc.pth'.format(modelNumber=modelNumber))
#load_checkpoint('output_mnist_2drecurr0/checkpoint_most_recent.pth') # MLR2.0 trained on emnist letters, digits, and fashion mnist

#print('Loading the classifiers')
'''clf_shapeS=load('classifier_output/ss.joblib')
clf_shapeC=load('classifier_output/sc.joblib')
clf_colorC=load('classifier_output/cc.joblib')
clf_colorS=load('classifier_output/cs.joblib')
'''
#write to a text file
#outputFile = open('outputFile.txt'.format(modelNumber),'w')

bs_testing = 1000     # number of images for testing. 20000 is the limit
shape_coeff = 1       #cofficient of the shape map
color_coeff = 1       #coefficient of the color map
location_coeff = 0    #Coefficient of Location map
l1_coeff = 1          #coefficient of layer 1
l2_coeff = 1          #coefficient of layer 2
shapeLabel_coeff= 1   #coefficient of the shape label
colorLabel_coeff = 1  #coefficient of the color label
location_coeff = 0  #coefficient of the color label

bpsize = 12000#00         #size of the binding pool
token_overlap =0.3
bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

normalize_fact_familiar=1
normalize_fact_novel=1


imgsize = 28
all_imgs = []

#number of repetions for statistical inference
hugepermnum=8000
bigpermnum = 500
smallpermnum = 100

Fig2aFlag = 0       #binding pool reconstructions   NOTWORKING
fig_new_loc = 0     # reconstruct retina images with digits in the location opposite of training
fig_loc_compare = 0 # compare retina images with digits in the same location as training and opposite location  
Fig2bFlag = 0   #novel objects stored and retrieved from memory, one at a time
Fig2btFlag =0     #novel objects stored and retrieved from memory, in tokens
Fig2cFlag = 0      #familiar objects stored and retrieved from memory, using tokens 
sampleflag = 0   #generate random objects from latents (plot working, not behaving as expected)
Fig2nFlag = 0
change_detect_flag = 0
change_detect_2_flag = 0
BP_std = 0
    
bindingtestFlag = 0  #simulating binding shape-color of two items  NOT WORKING

Tab1Flag_noencoding = 0 #classify reconstructions (no memory) NOT WORKINGy
Tab1Flag = 0 #             #classify binding pool memoriesNOT WORKING
Tab1SuppFlag = 0        #memory of labels (this is table 1 + Figure 2 in supplemental which includes the data in Figure 3)
Tab2Flag =0 #NOT WORKING
TabTwoColorFlag = 0  #NOT WORKING
TabTwoColorFlag1 = 0            #Cross correlations for familiar vs novel  #NOT WORKING
noveltyDetectionFlag=0  #detecting whether a stimulus is familiar or not  #NOT WORKING
latents_crossFlag = 0   #Cross correlations for familiar vs novel for when infromation is stored from the shape/color maps vs. L1. versus straight reconstructions
                        #This Figure is not included in the paper  #NOT WORKING

bs=100   # number of samples to extract from the dataset

#### generate some random samples (currently commented out due to cuda errors)  #NOT WORKING
if (sampleflag):
    zc=torch.randn(64,16).cuda()*1
    zs=torch.randn(64,16).cuda()*1
    with torch.no_grad():
        sample = vae.decoder_cropped(zs,zc,0).cuda()
        sample_c= vae.decoder_cropped(zs*0,zc,0).cuda()
        sample_s = vae.decoder_cropped(zs, zc*0, 0).cuda()
        sample=sample.view(64, 3, 28, 28)
        sample_c=sample_c.view(64, 3, 28, 28)
        sample_s=sample_s.view(64, 3, 28, 28)
        save_image(sample[0:8], 'output{num}/sample.png'.format(num=modelNumber))
        save_image(sample_c[0:8], 'output{num}/sample_color.png'.format(num=modelNumber))
        save_image(sample_s[0:8], 'output{num}/sample_shape.png'.format(num=modelNumber))

    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))
    test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True, num_workers=nw)


def cd_r_acc_vs_setsize(vae):
    def compute_dprime(no_change_vector, change_vector):
        no_change_vector = np.array(no_change_vector)
        change_vector = np.array(change_vector)
        
        # hit rate and false alarm rate
        hit_rate = np.mean(change_vector)
        false_alarm_rate = 1 - np.mean(no_change_vector)
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
        
        # z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)
        d_prime = z_hit - z_fa
        
        return d_prime

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
    
    def build_partial(input_tensor, n, out_x=1):
        x, b, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            for img_idx in range(n):
                output_tensor[batch_idx] += input_tensor[batch_idx, img_idx]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    vae.eval()
    max_set_size = 8 #8
    out_r = {0:[], 1:[]} #[]
    out_dprime = {0:[], 1:[]} # []
    threshold = {0:[], 1:[]} #[]
    setsize_range = [2,3,4, 6, 8] #range(2, max_set_size+1, 1)
    task = '_color'
    for t in range(0,2): # must start at 0
        for i in setsize_range:
            if t == 0:
                frame_count = 1
            else:
                frame_count = i
            
            with torch.no_grad():
                if t == 0:
                    original_t = torch.load(f'original_{i}{task}.pth').cpu()
                    change_t = torch.load(f'change_{i}{task}.pth').cpu()
                else:
                    original_t = torch.load(f'original_frames_{i}{task}.pth').cpu()
                    change_t = torch.load(f'change_{i}{task}.pth').cpu()

                print(len(original_t),len(change_t))
                r_lst0 = []
                r_lst1 = []
                no_change_detected = []
                change_detected = []
                for b in range(0,1): #20
                    print(i,b)
                    torch.cuda.empty_cache()
                    samples = 20
                    batch_id = b*samples
                    original = original_t[batch_id: batch_id+samples].cuda()
                    change = change_t[batch_id: batch_id+samples].cuda()
                    #original_frames = torch.load(f'original_frames_{i}.pth').cuda()
                    batch_size = original.size(0)
                    #print(original_frames.size())

                    # store block arrays in BP via L1
                    l1_original = []
                    for n in range(len(original)):
                        l1_act, l2_act, shape_act, color_act, location_act = activations(original[n].view(frame_count,3,28,28))
                        l1_original += [l1_act]
                    #l1_change, l2_act, shape_act, color_act, location_act = activations(change)
                    bp_original_l1 = []
                    bp_change_l1 = []
                    bp_junk = torch.zeros(frame_count,1).cuda()
                    for n in range(batch_size):
                        
                        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original[n].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,0, 0,0,1,0,frame_count,normalize_fact_novel)
                        shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original[1].view(frame_count,-1), bp_junk, shape_act,color_act,bp_junk,frame_count,normalize_fact_novel)
                        #BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_change[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                        #shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_change = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_change[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                        if t == 0:
                            bp_original_l1 += [BP_layerI_original]
                        else:
                            bp_original_l1 += [BP_layerI_original] #[[shape_out_all, color_out_all]]
                        #bp_change_l1 += [BP_layerI_change]

                    #bp_original_l1 = torch.cat(bp_original_l1, dim=0)
                    #bp_change_l1 = torch.cat(bp_change_l1, dim=0)
                    original_BP = []
                    for n in range(len(original)):
                        if t == 0:
                            recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1[n].view(frame_count,-1),BP_layer2_out,3, 'skip_cropped')
                        else:
                            recon = recon, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1[n].view(frame_count,-1),BP_layer2_out,3, 'skip_cropped') #vae.decoder_cropped(bp_original_l1[n][0].view(frame_count,-1),bp_original_l1[n][1].view(frame_count,-1),0)
                            recon = build_partial(recon.view(1,frame_count,3,28,28), len(recon))
                        original_BP += [recon]
                    
                    if t == 1:
                        original = build_partial(original, len(original[0]))
                        #change = build_partial(change, len(change[0]))

                    #change_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_change_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')

                    save_image(torch.cat([original[0].view(1,3,28,28), original_BP[0]],dim=0), f'comp_memory.png', nrow = 1, normalize=False)
                    print(len(original_BP))
                    for j in range(len(original)):
                        #x, y, z = original[i].cpu().detach().view(1,-1), original_BP[i].cpu().detach().view(1,-1), change_BP[i].cpu().detach().view(1,-1)
                        x, y, z = original[j], original_BP[j].view(3,28,28), change[j]
                        r_original = compute_correlation(y,x) #cosine_similarity(x,y) # 
                        r_change = compute_correlation(y,z)#cosine_similarity(x,z) #
                        
                        r_lst0 += [r_original]
                        r_lst1 += [r_change]
                    
                    
                    del original
                    del change
                    del original_BP
                    del l1_original 
                    del l2_act 
                    del shape_act 
                    del color_act
                    del location_act
                    gc.collect()
                    torch.cuda.empty_cache()

            print('computing threshold')    
            avg_r0 = sum(r_lst0)/(len(r_lst0))
            avg_r1 = sum(r_lst1)/(len(r_lst1))
            out_r[t] += [[avg_r0.item(), avg_r1.item()]]
            #print(out_r[t])

            c_threshold = (avg_r0.item() + avg_r1.item())/2
            
            for l in range(len(r_lst0)):
                r_original = r_lst0[l]
                r_change = r_lst1[l]

                if r_original > c_threshold:
                    no_change_detected += [1]
                else:
                    no_change_detected += [0]

                if r_change <= c_threshold:
                    change_detected += [1]
                else:
                    change_detected += [0]

            out_dprime[t] += [compute_dprime(no_change_detected, change_detected)]
            threshold[t] += [c_threshold]
    print(out_r[0])
    torch.save([out_r[0][i][0] for i in range(len(out_r[0]))], 'location_change_detect_r.pth')
    plt.plot(setsize_range, [out_r[0][i][0] for i in range(len(out_r[0]))], label='no change')
    plt.plot(setsize_range, [out_r[0][i][1] for i in range(len(out_r[0]))], label='change')
    plt.plot(setsize_range, threshold[0], label='threshold')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'color change detection, {batch_size*2} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect{task}.png')
    plt.close()

    plt.plot(setsize_range, [out_r[1][i][0] for i in range(len(out_r[1]))], label='no change')
    plt.plot(setsize_range, [out_r[1][i][1] for i in range(len(out_r[1]))], label='change')
    plt.plot(setsize_range, threshold[1], label='threshold, compositional')
    plt.xlabel('set size')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'color change detection compositonal memory, {batch_size*2} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect_compositional{task}.png')
    plt.close()

    plt.plot(setsize_range, out_dprime[0], label=f'dprime for whole memory')
    plt.plot(setsize_range, out_dprime[1], label=f'dprime for compositional memory')
    plt.xlabel('set size')
    plt.ylabel('dprime')
    plt.legend()
    plt.title(f'color change dprime vs set size, {batch_size*2} trials, BP: {bpPortion}')
    plt.savefig(f'change_detect_accuracy{task}.png')

def cd_jiang_olson_chun_sim(vae):
    def compute_dprime(no_change_vector, change_vector):
        no_change_vector = np.array(no_change_vector)
        change_vector = np.array(change_vector)
        
        # hit rate and false alarm rate
        hit_rate = np.mean(change_vector)
        false_alarm_rate = 1 - np.mean(no_change_vector)
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
        
        # z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(false_alarm_rate)
        d_prime = z_hit - z_fa
        
        return d_prime

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

    def build_partial(input_tensor, n=4):
        x, batch_size, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            for img_idx in range(n):
                output_tensor[batch_idx] += input_tensor[batch_idx, img_idx]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

    def build_single(input_tensor):
        x, batch_size, channels, height, width = input_tensor.shape
        output_tensor = torch.zeros(x, channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)

        for batch_idx in range(x):
            output_tensor[batch_idx] += input_tensor[batch_idx, 0]

        # Clamp the output tensor to the range [0, 1]
        output_tensor = torch.clamp(output_tensor, min=0.0, max=1.0)

        return output_tensor

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

    vae.eval()
    max_set_size = 8 #8
    out_r = {'total':[], 'partial':[], 'single':[]}
    out_dprime = {'total':[], 'partial':[], 'single':[]}
    out_accuracy = {'total':[], 'partial':[], 'single':[]} # []
    threshold = {0:[], 1:[]} #[]
    setsize_range = range(8,9)
    threshold_data = torch.load('location_change_detect_r.pth')
    #print(threshold_data)

    for b in range(0,1): #20
        for i in setsize_range:
            torch.cuda.empty_cache()
            samples = 50
            batch_id = b*samples
            #original = torch.load(f'original_{i}.pth')[batch_id: batch_id+samples].cuda()
            #change = torch.load(f'change_{i}.pth')[batch_id: batch_id+samples].cuda()
            original_frames = torch.load(f'original_frames_{i}.pth')[batch_id: batch_id+samples].cuda()
            change_frames = torch.load(f'change_frames_{i}.pth')[batch_id: batch_id+samples].cuda()
            original = build_partial(original_frames, n=len(original_frames[0]))
            change = build_partial(change_frames, n=len(change_frames[0]))
            position_lst = torch.load(f'positions_{i}.pth')[batch_id: batch_id+samples]
            print(position_lst[0][0])
            batch_size = original.size(0)
            print(i, batch_size)
            original_partial = build_partial(original_frames)
            change_partial = build_partial(change_frames)
            original_single = build_single(original_frames)
            change_single = build_single(change_frames)
            print(original_partial.size())
            #save_image(torch.cat([original, change, original_partial,change_partial, original_single, change_single],dim=0), f'changedetect_color.png', nrow = batch_size, normalize=False)
            
            # store block arrays in BP via L1

            l1_original, l2_act, shape_act, color_act, location_act = activations(original)
            #l1_change, l2_act, shape_act, color_act, location_act = activations(change)
            bp_original_l1 = []
            bp_change_l1 = []
            bp_junk = torch.zeros(1,1).cuda()
            for n in range(batch_size):
                BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_original[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_original = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_original[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                #BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_change[n,:].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,0, 0,0,1,0,1,normalize_fact_novel)
                #shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_change = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings, l1_change[1].view(1,-1), bp_junk, bp_junk,bp_junk,bp_junk,1,normalize_fact_novel)
                
                bp_original_l1 += [BP_layerI_original]
                #bp_change_l1 += [BP_layerI_change]

            bp_original_l1 = torch.cat(bp_original_l1, dim=0)
            #bp_change_l1 = torch.cat(bp_change_l1, dim=0)
            original_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_original_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')
            #change_BP, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(bp_change_l1.view(batch_size,-1),BP_layer2_out,3, 'skip_cropped')
            #save_image(torch.cat([original, original_BP],dim=0), f'changedetect_location_bp.png', nrow = batch_size, normalize=False)


            r_lst0 = []
            r_lst1 = []
            r_lst2 = []
            r_lst3 = []
            r_lst4 = []
            r_lst5  = []
            no_change_detected = []
            change_detected = []
            no_change_detected_partial = []
            change_detected_partial = []
            no_change_detected_single = []
            change_detected_single = []
            for j in range(len(original)):
                #x, y, z = original[i].cpu().detach().view(1,-1), original_BP[i].cpu().detach().view(1,-1), change_BP[i].cpu().detach().view(1,-1)
                x, y, z = original[j], original_BP[j], change[j]
                r_original_total = compute_correlation(x,y) #cosine_similarity(x,y) # 
                r_change_total = compute_correlation(y,z)#cosine_similarity(x,z) #
                
                r_original_partial = compute_correlation(y,original_partial[j]) #cosine_similarity(x,y) # 
                r_change_partial = compute_correlation(y,change_partial[j])
                
                pos = position_lst[j][0]
                y = zero_outside_radius(y,pos[0],pos[1])
                save_image(torch.cat([x.view(1,3,28,28),original_BP[j].view(1,3,28,28),y.view(1,3,28,28), original_single[j].view(1,3,28,28)],dim=0), f'single_masked.png', nrow = 1, normalize=False)
                
                r_original_single = compute_correlation(y,original_single[j]) #cosine_similarity(x,y) # 
                r_change_single = compute_correlation(y,change_single[j])
            
                r_lst0 += [r_original_total]
                r_lst1 += [r_change_total]

                
                r_lst2 += [r_original_partial]
                r_lst3 += [r_change_partial]
                
                # compute r only in area around probe object location
                r_lst4 += [r_original_single]
                r_lst5 += [r_change_single]
            
            avg_r0 = sum(r_lst0)/(len(r_lst0))
            avg_r1 = sum(r_lst1)/(len(r_lst1))
            avg_r2 = sum(r_lst2)/(len(r_lst2))
            avg_r3 = sum(r_lst3)/(len(r_lst3))
            avg_r4 = sum(r_lst4)/(len(r_lst4))
            avg_r5 = sum(r_lst5)/(len(r_lst5))

            if len(out_r['total']) == 0:
                out_r['total'] = [avg_r0.item(), avg_r1.item()] 
                out_r['partial'] = [avg_r2.item(), avg_r3.item()]
                out_r['single'] = [avg_r4.item(), avg_r5.item()]
            else:
                out_r['total'] = [(avg_r0.item()+out_r['total'][0])/2, (avg_r1.item()+out_r['total'][1])/2] 
                out_r['partial'] = [(avg_r2.item()+out_r['partial'][0])/2, (avg_r3.item()+out_r['partial'][1])/2]
                out_r['single'] = [(avg_r4.item()+out_r['single'][0])/2, (avg_r5.item()+out_r['single'][1])/2]
            
            threshold_scalar = 0.9
            c_threshold_total = threshold_data[7] #(avg_r0.item() + avg_r1.item())/2
            c_threshold_partial = threshold_data[3] * threshold_scalar #(avg_r2.item() + avg_r3.item())/2
            c_threshold_single = threshold_data[0] #(avg_r4.item() + avg_r5.item())/2
            
            for l in range(len(r_lst0)):
                r_original = r_lst0[l]
                r_change = r_lst1[l]

                if r_original > c_threshold_total:
                    no_change_detected += [1]
                else:
                    no_change_detected += [0]

                if r_change <= c_threshold_total:
                    change_detected += [1]
                else:
                    change_detected += [0]
            
            # partial
            for l in range(len(r_lst2)):
                r_original = r_lst2[l]
                r_change = r_lst3[l]

                if r_original > c_threshold_partial:
                    no_change_detected_partial += [1]
                else:
                    no_change_detected_partial += [0]

                if r_change <= c_threshold_partial:
                    change_detected_partial += [1]
                else:
                    change_detected_partial += [0]
            
            # single
            for l in range(len(r_lst4)):
                r_original = r_lst4[l]
                r_change = r_lst5[l]

                if r_original > c_threshold_single:
                    no_change_detected_single += [1]
                else:
                    no_change_detected_single += [0]

                if r_change <= c_threshold_single:
                    change_detected_single += [1]
                else:
                    change_detected_single += [0]

            out_dprime['total'] += [compute_dprime(no_change_detected, change_detected)]
            out_accuracy['total'] += [((sum(no_change_detected)/len(no_change_detected)) + (sum(change_detected)/len(change_detected)))/2]
            #threshold[t] += [c_threshold]

            out_dprime['partial'] += [compute_dprime(no_change_detected_partial, change_detected_partial)]
            out_accuracy['partial'] += [((sum(no_change_detected_partial)/len(no_change_detected_partial)) + (sum(change_detected_partial)/len(change_detected_partial)))/2]
            #threshold[t] += [c_threshold]

            out_dprime['single'] += [compute_dprime(no_change_detected_single, change_detected_single)]
            out_accuracy['single'] += [((sum(no_change_detected_single)/len(no_change_detected_single)) + (sum(change_detected_single)/len(change_detected_single)))/2]
            #threshold[t] += [c_threshold]
            # course 
    out_dprime['total'] = sum(out_dprime['total'])/(len(out_dprime['total']))
    out_dprime['partial'] = sum(out_dprime['partial'])/(len(out_dprime['partial']))
    out_dprime['single'] = sum(out_dprime['single'])/(len(out_dprime['single']))

    out_accuracy['total'] = sum(out_accuracy['total'])/(len(out_accuracy['total']))
    out_accuracy['partial'] = sum(out_accuracy['partial'])/(len(out_accuracy['partial']))
    out_accuracy['single'] = sum(out_accuracy['single'])/(len(out_accuracy['single']))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = bar_width/2

    # Plot the bars in pairs
    ax.bar(x[0], out_r['total'][0], width=bar_width, label = 'full, no change')
    ax.bar(x[0] + gap_width, out_r['total'][1], width=bar_width, label = 'full, change')
    ax.bar(x[1], out_r['partial'][0], width=bar_width, label = 'partial, no change')
    ax.bar(x[1] + gap_width, out_r['partial'][1], width=bar_width, label = 'partial, change')
    ax.bar(x[2], out_r['single'][0], width=bar_width, label = 'single, no change')
    ax.bar(x[2] + gap_width, out_r['single'][1], width=bar_width, label = 'single, change')
    ax.set_ylabel('r')
    ax.legend()
    ax.set_title('Correlation for location change detection, 50 trials')
    plt.savefig('change_detect_bar.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = 0.5

    # Plot the bars in pairs
    ax.bar(x[0], out_dprime['total'], width=bar_width, label='full context')
    ax.bar(x[1], out_dprime['partial'], width=bar_width, label='partial context')
    ax.bar(x[2], out_dprime['single'], width=bar_width, label='single context')
    ax.legend()
    ax.set_title('dprime for location change detection, 50 trials')
    ax.set_ylabel('dprime')
    plt.savefig('change_detect_accuracy_bar.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create x-axis ticks with gaps between pairs
    x = np.arange(3)
    bar_width = 0.4
    gap_width = 0.5

    # Plot the bars in pairs
    ax.bar(x[0], out_dprime['total'], width=bar_width, label='full context')
    ax.bar(x[1], out_dprime['partial'], width=bar_width, label='partial context')
    ax.bar(x[2], out_dprime['single'], width=bar_width, label='single context')
    ax.legend()
    ax.set_title('accuracy for location change detection, 50 trials')
    ax.set_ylabel('%')
    plt.savefig('change_detect_accuracy_bar_nodprime.png')

######################## Figure 2a #######################################################################################
#store items using both features, and separately color and shape (memory retrievals)


def fig_2a(vae, folder_path):
    print('generating figure 2a, reconstructions from the binding pool')

    numimg= 6
    bs=numimg #number of images to display in this figure
    nw=2
    bs_testing = numimg # 20000 is the limit
    train_loader_noSkip = Dataset('mnist',{'colorize':True}).get_loader(bs)
    test_loader_noSkip = Dataset('mnist',{'colorize':True},train=False).get_loader(bs)

    test_loader_smaller = test_loader_noSkip
    images, shapelabels = next(iter(test_loader_smaller))#peel off a large number of images
    #orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
    imgs = images.clone().cuda()

    #run them all through the encoder
    l1_act, l2_act, shape_act, color_act, location_act = vae.activations(imgs)  #get activations from this small set of images
    '''BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act[n,:].view(1,-1), l2_act[n,:].view(1,-1), shape_act[n,:].view(1,-1),color_act[n,:].view(1,-1),location_act[n,:].view(1,-1),0, 0,0,1,0,1,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1,normalize_fact_novel)
    '''
    '''def BPTokens_storage(bpsize, bpPortion,l1_act, l2_act, shape_act, color_act, location_act, shape_coeff, color_coeff, location_coeff, l1_coeff,l2_coeff, bs_testing, normalize_fact):
    '''
    BPOut_all, Tokenbindings_all = BPTokens_storage(bpsize, bpPortion, l1_act, l2_act, shape_act,color_act,location_act,shape_coeff, color_coeff, location_coeff, l1_coeff,l2_coeff,1,normalize_fact_novel)
        
    shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut_all, Tokenbindings_all,l1_act, l2_act, shape_act,color_act,location_act,1,normalize_fact_novel)

    
    #memory retrievals from Bottleneck storage
    bothRet = vae.decoder_cropped(shape_out_all, color_out_all,0, 0).cuda()  # memory retrieval from the bottleneck
    #shapeRet = vae.decoder_shape(shape_out_BP_shapeonly, color_out_BP_shapeonly , 0).cuda()  #memory retrieval from the shape map
    #colorRet = vae.decoder_color(shape_out_BP_coloronly, color_out_BP_coloronly, 0).cuda()  #memory retrieval from the color map
    shapeRet = bothRet
    colorRet = bothRet
    save_image(
        torch.cat([imgs[0: numimg].view(numimg, 3, 28, 28), bothRet[0: numimg].view(numimg, 3, 28, 28),
                   shapeRet[0: numimg].view(numimg, 3, 28, 28), colorRet[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_BP_bottleneck_.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )

    #memory retrievals when information was stored from L1 and L2
    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 1, 'noskip') #bp retrievals from layer 1
    BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 2, 'noskip') #bp retrievals from layer 2

    save_image(
        torch.cat([
                   BP_layer2_noskip[0: numimg].view(numimg, 3, 28, 28), BP_layer1_noskip[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_layer2_layer1.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )

def fig_loc_compare(vae, folder_path):
    vae.eval()
    bs = 15
    mnist_transforms = {'retina':True, 'colorize':True, 'scale':False, 'location_targets':{'left':[0,1,2,3,4],'right':[5,6,7,8,9]}}
    mnist_test_transforms = {'retina':True, 'colorize':True, 'scale':False, 'location_targets':{'right':[0,1,2,3,4],'left':[5,6,7,8,9]}}
    train_loader_noSkip = Dataset('mnist',mnist_transforms).get_loader(bs)
    test_loader_noSkip = Dataset('mnist',mnist_test_transforms, train=False).get_loader(bs)
    imgsize = 28
    numimg = 10
    
    dataiter_noSkip_test = iter(test_loader_noSkip)
    dataiter_noSkip_train = iter(train_loader_noSkip)
    #skipd = iter(train_loader_skip)
    #skip = skipd.next()
    #print(skip[0].size())
    #print(type(dataiter_noSkip_test))
    data_test = next(dataiter_noSkip_test)
    data_train = next(dataiter_noSkip_train)

    data = data_train[0].copy()

    data[0] = torch.cat((data_test[0][0], data_train[0][0]),dim=0) #.cuda()
    data[1] = torch.cat((data_test[0][1], data_train[0][1]),dim=0)
    data[2] = torch.cat((data_test[0][2], data_train[0][2]),dim=0)

    sample = data
    sample_size = 15
    #print(sample[0].size(),sample[1].size(),sample[2].size())
    with torch.no_grad():
        reconl, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'location') #location
        reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'retinal') #retina

    line1 = torch.ones((1,2)) * 0.5
    line2 = line1.view(1,1,1,2)
    line3 = line2.expand(sample_size, 3, retina_size, 2).cuda()
    line2 = line2.expand(sample_size*2, 3, retina_size, 2).cuda()
    #reconb = reconb['recon']

    shape_color_dim = retina_size + 2
    shape_color_dim1 = imgsize + 2
    sample = torch.cat((sample[0].cuda(),line2),dim = 3).cuda()
    reconb = torch.cat((reconb,line2.cuda()),dim = 3).cuda()
    shape_color_dim = retina_size + 2
    sample_test = sample[:sample_size] #torch.cat((sample[0][:sample_size],line3),dim = 3).cuda()
    sample_train = sample[sample_size:] # torch.cat((sample[0][sample_size:],line3),dim = 3).cuda()
    utils.save_image(
        torch.cat([sample_train.view(sample_size, 3, retina_size, shape_color_dim), reconb[sample_size:(2*sample_size)].view(sample_size, 3, retina_size, shape_color_dim), 
                   sample_test.view(sample_size, 3, retina_size, shape_color_dim), reconb[:(sample_size)].view(sample_size, 3, retina_size, shape_color_dim)], 0),
        f'{folder_path}figure_location_compare.png',
        nrow=sample_size, normalize=False)
    
    image_pil = Image.open(f'{folder_path}figure_location_compare.png')
    trained_label = "Trained Data"
    untrained_label = "Untrained Data"
    # Add trained and untrained labels to the image using PIL's Draw module
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()  # You can choose a different font or size

    # Trained data label at top left
    trained_label_position = (10, 10)  # Adjust the position of the text
    draw.text(trained_label_position, trained_label, fill=(255, 255, 255), font=font)

    # Untrained data label at bottom left
    image_width, image_height = image_pil.size
    untrained_label_position = (10, image_height//2)  # Adjust the position of the text
    draw.text(untrained_label_position, untrained_label, fill=(255, 255, 255), font=font)

    # Save the modified image with labels
    image_pil.save(f'{folder_path}figure_location_compare.png')

    print("Images with labels saved successfully.")
    return []

def fig_2n(vae, folder_path):
    print('bengali reconstructions')
    all_imgs = []
    imgsize = 28
    numimg = 7

    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        img_new = convert_tensor(Image.open(f'current_bengali/{i}_thick.png'))[0:3,:,:]
        #img_new = Colorize_func(img)   # Currently broken, but would add a color to each
        all_imgs.append(img_new)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3, imgsize, imgsize).cuda()
    output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(all_imgs,whichdecode='skip_cropped')
    z_img = vae.sampling(mu_shape,log_var_shape)
    recon_sample = vae.decoder_shape(z_img, 0, 0)
    out_img = torch.cat([imgs[0: numimg].view(numimg, 3, 28, imgsize),output,recon_sample],dim=0)
    utils.save_image(out_img,f'output{modelNumber}/bengali_recon.png',numimg)

def BP_std(vae, folder_path):
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
    #from dataset_builder import Colorize_specific
    all_imgs = []
    print('computing BP std correlations')
    numimg = 7
    vae.eval()
    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        #colorize_img = Colorize_specific(col=1)
        img = Image.open(f'current_bengali/{i}_thick.png')# Image.open(f'change_image_{i}.png') #
        img = img.resize((28, 28))
        #img_new = colorize_img(img)   # Currently broken, but would add a color to each
        img_new = convert_tensor(img)[0:3,:,:]
        #img_new[0] = torch.zeros_like(img_new[0])
        #img_new[2] = torch.zeros_like(img_new[1])

        all_imgs.append(img_new)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()
    location = torch.zeros(imgs.size()[0], vae.l_dim).cuda()
    location[0] = 1

    blank = torch.zeros(1,3,28,28).cuda()
    blank[:,:,]
    #push the images through the encoder
    l1_act, l2_act, shape_act, color_act, location_act = activations(imgs.view(-1,3,28,28), location)
    
    #now run them through the binding pool!
    #store the items and then retrive them, and do it separately for shape+color maps, then L1, then L2. 
    #first store and retrieve the shape, color and location maps
    r_lst = []
    std_range = []
    with torch.no_grad():
        for c in range(0,21):
            std = c * 0.05
            imgmatrixL1skip  = torch.empty((0,3,28,28)).cuda()
            for n in range (0,numimg):
                    # reconstruct directly from activation
                #recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped')
                
                #now store/retrieve from L1
                BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act[n,:].view(1,-1), l2_act[n,:].view(1,-1), shape_act[n,:].view(1,-1),color_act[n,:].view(1,-1),location_act[n,:].view(1,-1),0, 0,0,1,0,1,normalize_fact_novel,std)
                shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1,normalize_fact_novel)
                #print(BP_layerI_out.size())
                # reconstruct  from BP version of layer 1, run through the skip
                #BP_layerI_out = l1_act[n,:].view(1,-1) #remove
                #BP_layerI_out = vae.sparse_relu(BP_layerI_out*(1/2))

                BP_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out.view(1,-1),BP_layerI_out,3, 'skip_cropped')

                imgmatrixL1skip = torch.cat([imgmatrixL1skip,BP_layer1_skip])
            
            r = compute_correlation(imgs.view(-1,3,28,28), imgmatrixL1skip.view(-1,3,28,28))
            r_lst += [r.item()]
            std_range += [std]
    plt.plot(std_range, r_lst)
    plt.xlabel('BP std')
    plt.ylabel('r')
    plt.title(f'Novel recon correlation vs BP std')
    plt.savefig(f'BP_std_corr.png')
    plt.close()

def fig_2b(vae, folder_path):
    #from dataset_builder import Colorize_specific
    all_imgs = []
    print('generating Figure 2b, Novel characters retrieved from memory of L1 and Bottleneck')
    retina_size = 100
    imgsize = 28
    numimg = 7
    vae.eval()
    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        #colorize_img = Colorize_specific(col=1)
        img = Image.open(f'current_bengali/{i}_thick.png')# Image.open(f'change_image_{i}.png') #
        img = img.resize((28, 28))
        #img_new = colorize_img(img)   # Currently broken, but would add a color to each
        img_new = convert_tensor(img)[0:3,:,:]
        #img_new[0] = torch.zeros_like(img_new[0])
        #img_new[2] = torch.zeros_like(img_new[1])

        all_imgs.append(img_new)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()
    location = torch.zeros(imgs.size()[0], vae.l_dim).cuda()
    location[0] = 1

    blank = torch.zeros(1,3,28,28).cuda()
    blank[:,:,]
    #push the images through the encoder
    l1_act, l2_act, shape_act, color_act, location_act = vae.activations(imgs.view(-1,3,28,28), location)
    '''for i in range(len(l1_act)):
        z = len(l1_act[i])//2
        l1_act[i,:z]  *= -1'''

    imgmatrixL1skip  = torch.empty((0,3,28,28)).cuda()
    imgmatrixL1noskip  = torch.empty((0,3,28,28)).cuda()
    imgmatrixMap  = torch.empty((0,3,28,28)).cuda()
    
    #now run them through the binding pool!
    #store the items and then retrive them, and do it separately for shape+color maps, then L1, then L2. 
    #first store and retrieve the shape, color and location maps
    with torch.no_grad():
        for n in range (0,numimg):
                # reconstruct directly from activation
            #recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped')
            
            #now store/retrieve from L1
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act[n,:].view(1,-1), l2_act[n,:].view(1,-1), shape_act[n,:].view(1,-1),color_act[n,:].view(1,-1),location_act[n,:].view(1,-1),0, 0,0,1,0,1,normalize_fact_novel)
            shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1,normalize_fact_novel)
            #print(BP_layerI_out.size())
            # reconstruct  from BP version of layer 1, run through the skip
            #BP_layerI_out = l1_act[n,:].view(1,-1) #remove
            #BP_layerI_out = vae.sparse_relu(BP_layerI_out*(1/2))
            '''for i in range(len(BP_layerI_out)):
                z = len(BP_layerI_out[i])//2
                BP_layerI_out[i,:z]  *= -1'''

            BP_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out.view(1,-1),BP_layer2_out,3, 'skip_cropped')

            # reconstruct  from BP version of layer 1, run through the bottleneck
            BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out.view(1,-1),BP_layer2_out, 3, 'cropped')
            
            BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act[n,:].view(1,-1), l2_act[n,:].view(1,-1), shape_act[n,:].view(1,-1),color_act[n,:].view(1,-1),location_act[n,:].view(1,-1),1, 1,0,0,0,1,normalize_fact_novel)
            shape_out_BP, color_out_BP, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1,normalize_fact_novel)
            
            #reconstruct from BP version of the shape and color maps
            retrievals = vae.decoder_cropped(shape_out_BP, color_out_BP,0,0).cuda()

            imgmatrixL1skip = torch.cat([imgmatrixL1skip,BP_layer1_skip])
            imgmatrixL1noskip = torch.cat([imgmatrixL1noskip,BP_layer1_noskip])
            imgmatrixMap= torch.cat([imgmatrixMap,retrievals])

    #save an image showing:  original images, reconstructions directly from L1,  from L1 BP, from L1 BP through bottleneck, from maps BP
    save_image(torch.cat([imgs[0: numimg].view(numimg, 3, 28, imgsize), imgmatrixL1skip, imgmatrixL1noskip, imgmatrixMap], 0),'output{num}/figure2b.png'.format(num=modelNumber),
            nrow=numimg,            normalize=False,)  

def fig_2bt(vae, folder_path):
    all_imgs = []
    recon = list()
    print('generating Figure 2bt, Novel characters retrieved from memory of L1 and Bottleneck using Tokens')
    retina_size = 100
    imgsize = 28
    numimg = 7  #how many objects will we use here?
    vae.eval()

    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        img_new = convert_tensor(Image.open(f'current_bengali/{i}_thick.png'))[0:3,:,:]
        all_imgs.append(img_new)

    #all_imgs is a list of length 3, each of which is a 3x28x28
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()   #dimensions are R+G+B + # pixels
    imgs[0], imgs[6] = imgs[6], imgs[0]
    imgmatrix = imgs.view(numimg,3,28,28)
    #push the images through the model
    l1_act, l2_act, shape_act, color_act, location_act = activations(imgs.view(-1,3,28,28))
    emptyshape = torch.empty((1,3,28,28)).cuda()
    # store 1 -> numimg items
    for i in range(len(l1_act)):
        z = len(l1_act[i])//2
        l1_act[i,:z]  *= -1
    for n in range(1,numimg+1):
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,0, 0,0,1,0,n,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,n,normalize_fact_novel)
               
        plt.close()

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the first distribution (post BP)
        ax2.hist(l1_out_all[0].cpu().detach())
        ax2.set_title('BP')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        for i in range(len(l1_out_all)):
            z = len(l1_out_all[i])//2
            l1_out_all[i,:z]  *= -1
        #l1_out_all = vae.sparse_relu(l1_out_all*(1/2)) 
        # Plot the second distribution (pre BP)
        ax1.hist(l1_act[0].cpu().detach()) 
        ax1.set_title('pre BP')
        #ax1.set_xlabel('Value')
        #ax1.set_ylabel('Frequency')

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig('skipconhist.png')
        plt.close()
        recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_out_all.view(n,-1), l2_act, 3, 'skip_cropped')
        #recon_layer1_skip= vae.decoder_skip_cropped(0,0,0,l1_out_all)      
        imgmatrix= torch.cat([imgmatrix,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrix= torch.cat([imgmatrix,emptyshape*0],0)

    save_image(imgmatrix,'output{num}/figure2bt.png'.format(num=modelNumber),    nrow=numimg, normalize=False,   )



def fig_2c(vae, folder_path):
    vae.eval()
    print('generating Figure 2c, Familiar characters retrieved from Bottleneck using Tokens')
    retina_size = 100
    reconMap = list()
    reconL1 = list()
    imgsize = 28
    numimg = 7  #how many objects will we use here?
    #torch.set_default_dtype(torch.float64)
    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    test_loader_noSkip= Dataset('mnist',{'colorize':True,'retina':True}, train=False).get_loader(numimg)
    
    #Code showing the data loader for how the model was trained, empty dict in 3rd param is for any color:
    '''train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            {},True,{'right':list(range(0,5)),'left':list(range(5,10))}) '''    

    dataiter_noSkip = iter(test_loader_noSkip)
    data = next(dataiter_noSkip)
    data = data[0] #.cuda()
    
    sample_data = data
    sample_size = numimg
    sample_data[0] = sample_data[0][:sample_size]
    sample_data[1] = sample_data[1][:sample_size]
    sample = sample_data
    
    
    #push the images through the model
    reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,mu_scale,log_var_scale = vae(sample, 'cropped')
    l1_act, l2_act, shape_act, color_act, location_act = vae.activations(sample[1].view(-1,3,28,28).cuda())
    
    shape_act = vae.sampling(mu_shape, log_var_shape).cuda()
    color_act = vae.sampling(mu_color, log_var_color).cuda()
    reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped') 

    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = torch.cat([sample[1].view(numimg,3,28,28).cuda(), reconb],0)
    imgmatrixL1 = torch.cat([sample[1].view(numimg,3,28,28).cuda(), reconskip],0)
    shape_act_in = shape_act
    # store 1 -> numimg items
    for n in range(1,numimg+1):
        #Store and retrieve the map versions
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1, 1,0,0,0,n,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,n,normalize_fact_novel)
        z = torch.randn(numimg-n,8).cuda()
        #shape_out_all = torch.cat([shape_out_all,z],0)
        #color_out_all = torch.cat([color_out_all,z],0)
        #shape_out_all = shape_act[:n]
        #color_out_all = color_act[:n] #torch.cat([color_out_all,z],0)
        #shape_out_all = torch.cat([shape_out_all,z],0)
        #color_out_all = torch.cat([color_out_all,z],0)
        #retrievals = []
        #for i in range(n):
         #   print(color_out_all[i].view(-1,vae.z_dim).size())
         #   retrievals += [vae.decoder_cropped(shape_out_all[i].view(-1,vae.z_dim), color_out_all[i].view(-1,vae.z_dim),0,0).cuda()]
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()
        #retrievals = retrievals[:n]
        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,0, 0,0,1,0,n,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,n,normalize_fact_novel)
        #l1_out_all=l1_act[:n] #remove
        recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_out_all.view(n,-1), l2_act, 3, 'skip_cropped')        
        
        #imgmatrixMap= torch.cat([imgmatrixMap] + retrievals,0)
        
        imgmatrixMap= torch.cat([imgmatrixMap, retrievals],0)
        imgmatrixL1= torch.cat([imgmatrixL1,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrixMap= torch.cat([imgmatrixMap,emptyshape*0],0)
            imgmatrixL1= torch.cat([imgmatrixL1,emptyshape*0],0)
 
    save_image(imgmatrixL1, 'output{num}/figure2cL1.png'.format(num=modelNumber),  nrow=numimg,        normalize=False) #range=(-1, 1))
    save_image(imgmatrixMap, 'output{num}/figure2cMap.png'.format(num=modelNumber),  nrow=numimg,        normalize=False) #,range=(-1, 1))
        

###################Table 2##################################################
if Tab2Flag ==1:

    numModels=10

    print('Tab2 loss of quality of familiar vs novel items using correlation')

    setSizes=[1,2,3,4] #number of tokens


    familiar_corr_all=list()
    familiar_corr_all_se=list()
    novel_corr_all=list()
    novel_corr_all_se=list()

    familiar_skip_all=list()
    familiar_skip_all_se=list()

    novel_BN_all=list()
    novel_BN_all_se = list()

    perms = bigpermnum#number of times it repeats storing/retrieval



    for numItems in setSizes:

        familiar_corr_models = list()
        novel_corr_models = list()
        familiar_skip_models=list()
        novel_BN_models=list()

        print('SetSize {num}'.format(num=numItems))

        for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10

            load_checkpoint(
                'output{modelNumber}/checkpoint_threeloss_singlegrad50.pth'.format(modelNumber=modelNumber))

            # reset the data set for each set size
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=nw)

            # This function is in tokens_capacity.py

            familiar_corrValues= storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0,1)

            familiar_corrValues_skip = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                      test_loader_smaller, 'fam', 1, 1)


            novel_corrValues = storeretrieve_crosscorrelation_test(numItems, perms, bpsize,
                                                                                             bpPortion, shape_coeff,
                                                                                             color_coeff,
                                                                                             normalize_fact_familiar,
                                                                                             normalize_fact_novel,
                                                                                             modelNumber,
                                                                                             test_loader_smaller, 'nov',
                                                                                             1,1)
            novel_corrValues_BN = storeretrieve_crosscorrelation_test(numItems, perms, bpsize,
                                                                   bpPortion, shape_coeff,
                                                                   color_coeff,
                                                                   normalize_fact_familiar,
                                                                   normalize_fact_novel,
                                                                   modelNumber,
                                                                   test_loader_smaller, 'nov',
                                                                   0, 1)


            familiar_corr_models.append(familiar_corrValues)
            familiar_skip_models.append(familiar_corrValues_skip)
            novel_corr_models.append(novel_corrValues)
            novel_BN_models.append(novel_corrValues_BN)




        familiar_corr_models_all=np.array(familiar_corr_models).reshape(-1,1)
        novel_corr_models_all = np.array(novel_corr_models).reshape(1, -1)

        familiar_skip_models_all=np.array(familiar_skip_models).reshape(1,-1)
        novel_BN_models_all=np.array(novel_BN_models).reshape(1,-1)





        familiar_corr_all.append(np.mean(familiar_corr_models_all))
        familiar_corr_all_se.append(np.std(familiar_corr_models_all)/math.sqrt(numModels))


        novel_corr_all.append(np.mean( novel_corr_models_all))
        novel_corr_all_se.append(np.std(novel_corr_models_all)/math.sqrt(numModels))

        familiar_skip_all.append(np.mean(familiar_skip_models_all))
        familiar_skip_all_se.append(np.std(familiar_skip_models_all)/math.sqrt(numModels))

        novel_BN_all.append(np.mean(novel_BN_models_all))
        novel_BN_all_se.append(np.std(novel_BN_models_all)/math.sqrt(numModels))

    #the mean correlation value between input and recontructed images for familiar and novel stimuli
    outputFile.write('Familiar correlation\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE  {2:.3g}\n'.format(setSizes[i],familiar_corr_all[i],familiar_corr_all_se[i]))



    outputFile.write('\nfNovel correlation\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], novel_corr_all[i], novel_corr_all_se[i]))

    outputFile.write('\nfamiliar correlation vis skip \n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], familiar_skip_all[i], familiar_skip_all_se[i]))

    outputFile.write('\nnovel correlation via BN \n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], novel_BN_all[i], novel_BN_all_se[i]))

    #This part (not included in the paper) visualizes the cross correlation between novel shapes retrieved from the skip and familiar shapes retrived from the BN
    plt.figure()
    familiar_corr_all=np.array(familiar_corr_all)

    novel_corr_all=np.array(novel_corr_all)

    plt.errorbar(setSizes,familiar_corr_all,yerr=familiar_corr_all_se, fmt='o',markersize=3)
    plt.errorbar(setSizes, novel_corr_all, yerr=novel_corr_all_se, fmt='o', markersize=3)


    plt.axis([0,6, 0, 1])
    plt.xticks(np.arange(0,6,1))
    plt.show()

#############################################
if latents_crossFlag ==1:

    numModels=10

    print('cross correlations for familiar items when reconstructed and when retrived from BN or L1+skip ')

    setSizes=[1,2,3,4] #number of tokens

    noskip_recon_mean=list()
    noskip_recon_se=list()
    noskip_ret_mean=list()
    noskip_ret_se=list()

    skip_recon_mean=list()
    skip_recon_se=list()

    skip_ret_mean=list()
    skip_ret_se=list()

    perms = bigpermnum #number of times it repeats storing/retrieval

    for numItems in setSizes:

        noskip_Reconmodels=list()
        noskip_Retmodels=list()

        skip_Reconmodels=list()
        skip_Retmodels=list()

        print('SetSize {num}'.format(num=numItems))

        for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10

            load_checkpoint(
                'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

            # reset the data set for each set size
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=nw)

            # This function is in tokens_capacity.py
            #familiar items reconstrcuted via BN with no memory
            noskip_noMem= storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0, 0)
            # familiar items retrieved via BN
            noskip_Mem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0, 1)
            #recon from L1
            skip_noMem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                     test_loader_smaller, 'fam', 1, 0)
             #retrieve from L1 +skip
            skip_Mem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                      test_loader_smaller, 'fam', 1, 1)


            noskip_Reconmodels.append(noskip_noMem)
            noskip_Retmodels.append(noskip_Mem)

            skip_Reconmodels.append(skip_noMem)
            skip_Retmodels.append(skip_Mem)




        noskip_Reconmodels_all=np.array(noskip_Reconmodels).reshape(-1,1)
        noskip_Retmodels_all=np.array(noskip_Retmodels).reshape(-1,1)
        skip_Reconmodels_all = np.array(skip_Reconmodels).reshape(1, -1)
        skip_Retmodels_all=np.array(skip_Retmodels).reshape(1,-1)




        noskip_recon_mean.append(np.mean(noskip_Reconmodels_all))
        noskip_recon_se.append(np.std(noskip_Reconmodels_all)/math.sqrt(numModels))

        noskip_ret_mean.append(np.mean(noskip_Retmodels_all))
        noskip_ret_se.append(np.std(noskip_Retmodels_all) / math.sqrt(numModels))

        skip_recon_mean.append(np.mean(skip_Reconmodels_all))
        skip_recon_se.append(np.std(skip_Reconmodels_all) / math.sqrt(numModels))

        skip_ret_mean.append(np.mean(skip_Retmodels_all))
        skip_ret_se.append(np.std(skip_Retmodels_all) / math.sqrt(numModels))






    #the mean correlation value between input and recontructed images for familiar and novel stimuli
    outputFile.write('correlation for recons from BN\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE  {2:.3g}\n'.format(setSizes[i],noskip_recon_mean[i],noskip_recon_se[i]))

    outputFile.write('\nCorrelation for retrievals from BN\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i],noskip_ret_mean[i],noskip_ret_se[i]))

    outputFile.write('\ncorrelation for recons from skip\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], skip_recon_mean[i], skip_recon_se[i]))

    outputFile.write('\ncorrelation for retrievals from skip\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], skip_ret_mean[i], skip_ret_se[i]))

    plt.figure()

    correlations=np.array([skip_recon_mean,noskip_recon_mean, skip_ret_mean,noskip_ret_mean]).squeeze()

    corr_se=np.array([skip_recon_se,noskip_recon_se, skip_ret_se,noskip_ret_se]).squeeze()


    fig, ax = plt.subplots()
    pos=np.array([1,2,3,4])

    ax.bar(pos, correlations, yerr=corr_se, width=.4, alpha=.6, ecolor='black', color=['blue', 'blue', 'red', 'red'])


    plt.show()



########################  Ability to extract the correct token from a shape-only stimulus

if bindingtestFlag ==1:
    numModels=10
    perms = bigpermnum
    correctToken=np.tile(0.0,numModels)
    correctToken_diff=np.tile(0.0,numModels)
    accuracyColor=np.tile(0.0,numModels)
    accuracyColor_diff=np.tile(0.0,numModels)
    accuracyShape=np.tile(0.0,numModels)
    accuracyShape_diff=np.tile(0.0,numModels)

    for modelNumber in range(1,numModels+1):


        print('testing binding cue retrieval')



        # grey shape cue binding accuracy for only two items when the items are the same (e.g. two 3's ).


        bs_testing = 2
        correctToken[modelNumber-1],accuracyColor[modelNumber-1],accuracyShape[modelNumber-1] = binding_cue(bs_testing, perms, bpsize, bpPortion, shape_coeff, color_coeff, 'same',
                                                modelNumber)


        # grey shape cue binding accuracy for only two items when the two items are different
        correctToken_diff[modelNumber-1],accuracyColor_diff[modelNumber-1] ,accuracyShape_diff[modelNumber-1] = binding_cue(bs_testing, perms, bpsize, bpPortion, shape_coeff, color_coeff
                                                         , 'diff', modelNumber)


    correctToekn_all= correctToken.mean()
    SD=correctToken.std()

    correctToekn_diff_all=correctToken_diff.mean()
    SD_diff=correctToken_diff.std()
    accuracyColor_all=accuracyColor.mean()
    SD_color= accuracyColor.std()
    accuracyColor_diff_all=accuracyColor_diff.mean()
    SD_color_diff=accuracyColor_diff.std()

    accuracyShape_all=accuracyShape.mean()
    SD_shape= accuracyShape.std()
    accuracyShape_diff_all=accuracyShape_diff.mean()
    SD_shape_diff=accuracyShape_diff.std()



    outputFile.write('the correct retrieved token for same shapes condition is: {num} and SD is {sd}'.format(num=correctToekn_all, sd=SD))
    outputFile.write('\n the correct retrieved color for same shapes condition is: {num} and SD is {sd}'.format(num=accuracyColor_all, sd=SD_color))
    outputFile.write('\n the correct retrieved shape for same shapes condition is: {num} and SD is {sd}'.format(num=accuracyShape_all, sd=SD_shape))

    outputFile.write(
        '\n the correct retrieved token for different shapes condition is: {num} and SD is {sd}'.format(num=correctToekn_diff_all, sd=SD_diff))
    outputFile.write(
        '\n the correct retrieved color for different shapes condition is: {num} and SD is {sd}'.format(num=accuracyColor_diff_all, sd=SD_color_diff))
    outputFile.write(
        '\n the correct retrieved shape for different shapes condition is: {num} and SD is {sd}'.format(num=accuracyShape_diff_all, sd=SD_shape_diff))










#############Table 1 for the no memmory condition#####################
numModels = 1

perms=100

if Tab1Flag_noencoding == 1:

    print('Table 1 shape labels predicted by the classifier before encoded in memory')


    SSreport = np.tile(0.0,[perms,numModels])
    SCreport = np.tile(0.0,[perms,numModels])
    CCreport = np.tile(0.0,[perms,numModels])
    CSreport = np.tile(0.0,[perms,numModels])



    for temp in range(1,numModels +1):  # which model should be run, this can be 1 through 10

        modelNumber = 5
        load_checkpoint('output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

        print('doing model {0} for Table 1'.format(modelNumber))
        clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
        clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
        clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
        clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))

        for rep in range(0,perms):

           pred_cc, pred_cs, CCreport[rep,modelNumber - 1], CSreport[rep,modelNumber - 1] = classifier_color_test('noskip',
                                                                                                       clf_colorC,
                                                                                                       clf_colorS)

           pred_ss, pred_sc, SSreport[rep,modelNumber-1], SCreport[rep,modelNumber-1] = classifier_shape_test('noskip', clf_shapeS, clf_shapeC)


    print(CCreport)
    CCreport=CCreport.reshape(1,-1)
    CSreport=CSreport.reshape(1,-1)
    SSreport=SSreport.reshape(1,-1)
    SCreport=SCreport.reshape(1,-1)


    outputFile.write('Table 1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g}\n'.format(SSreport.mean(),SSreport.std()/math.sqrt(numModels*perms), SCreport.mean(),  SCreport.std()/math.sqrt(numModels) ))
    outputFile.write('Table 1, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g}\n'.format(CCreport.mean(),CCreport.std()/math.sqrt(numModels*perms), CSreport.mean(),  CSreport.std()/math.sqrt(numModels)))


########################## Table 1 for memory conditions ######################################################################

if Tab1Flag == 1:
    numModels=1
    perms=1
    SSreport_both = np.tile(0.0, [perms,numModels])
    SCreport_both = np.tile(0.0, [perms,numModels])
    CCreport_both = np.tile(0.0, [perms,numModels])
    CSreport_both = np.tile(0.0, [perms,numModels])
    SSreport_shape = np.tile(0.0, [perms,numModels])
    SCreport_shape = np.tile(0.0, [perms,numModels])
    CCreport_shape = np.tile(0.0, [perms,numModels])
    CSreport_shape = np.tile(0.0, [perms,numModels])
    SSreport_color = np.tile(0.0, [perms,numModels])
    SCreport_color = np.tile(0.0, [perms,numModels])
    CCreport_color = np.tile(0.0, [perms,numModels])
    CSreport_color = np.tile(0.0, [perms,numModels])
    SSreport_l1 = np.tile(0.0, [perms,numModels])
    SCreport_l1= np.tile(0.0, [perms,numModels])
    CCreport_l1 = np.tile(0.0, [perms,numModels])
    CSreport_l1 = np.tile(0.0, [perms,numModels])
    SSreport_l2 = np.tile(0.0, [perms,numModels])
    SCreport_l2 = np.tile(0.0, [perms,numModels])
    CCreport_l2 = np.tile(0.0, [perms,numModels])
    CSreport_l2 = np.tile(0.0, [perms,numModels])


    for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
        load_checkpoint(
            'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

        print('doing model {0} for Table 1'.format(modelNumber))
        clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
        clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
        clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
        clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


        print('Doing Table 1')

        for rep in range(0,perms):
            numcolors = 0

            colorlabels = thecolorlabels(test_dataset)

            bs_testing = 1000  # 20000 is the limit
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True,
                                                          num_workers=nw)
            colorlabels = colorlabels[0:bs_testing]

            images, shapelabels = next(iter(test_loader_smaller))  # peel off a large number of images
            orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
            imgs = orig_imgs.clone()

            # run them all through the encoder
            l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

            # now store and retrieve them from the BP
            BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                         shape_act, color_act,
                                                                                         shape_coeff, color_coeff, 1, 1,
                                                                                         normalize_fact_familiar)
            BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                                   l2_act, shape_act,
                                                                                                   color_act,
                                                                                                   shape_coeff, 0, 0, 0,
                                                                                                   normalize_fact_familiar)
            BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                                   l2_act, shape_act,
                                                                                                   color_act, 0,
                                                                                                   color_coeff, 0, 0,
                                                                                                   normalize_fact_familiar)

            BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                        shape_act, color_act, 0, 0,
                                                                                        l1_coeff, 0,
                                                                                        normalize_fact_familiar)

            BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                        shape_act, color_act, 0, 0, 0,
                                                                                        l2_coeff,
                                                                                        normalize_fact_familiar)

           # Table 1: classifier accuracy for shape and color for memory retrievals
            print('classifiers accuracy for memory retrievals of BN_both for Table 1')
            pred_ss, pred_sc, SSreport_both[rep,modelNumber-1], SCreport_both[rep,modelNumber-1] = classifier_shapemap_test_imgs(shape_out_BP_both, shapelabels,
                                                                             colorlabels, bs_testing, clf_shapeS,
                                                                             clf_shapeC)
            pred_cc, pred_cs, CCreport_both[rep,modelNumber-1], CSreport_both[rep,modelNumber-1]= classifier_colormap_test_imgs(color_out_BP_both, shapelabels,
                                                                             colorlabels, bs_testing, clf_colorC,clf_colorS)



            # Table 1: classifier accuracy for shape and color for memory retrievals
            print('classifiers accuracy for memory retrievals of BN_shapeonly for Table 1')
            pred_ss, pred_sc, SSreport_shape[rep,modelNumber - 1], SCreport_shape[
            rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_shapeonly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_shape[rep,modelNumber - 1], CSreport_shape[
            rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_shapeonly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_colorC, clf_colorS)

            print('classifiers accuracy for memory retrievals of BN_coloronly for Table 1')
            pred_ss, pred_sc, SSreport_color[rep,modelNumber - 1], SCreport_color[
            rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_coloronly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_color[rep,modelNumber - 1], CSreport_color[
            rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_coloronly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_colorC, clf_colorS)


            BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                                BP_layer2_out, 1,
                                                                                                'noskip')  # bp retrievals from layer 1
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            # Table 1 (memory retrievals from L1)
            print('classifiers accuracy for L1 ')
            pred_ss, pred_sc, SSreport_l1[rep,modelNumber - 1], SCreport_l1[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_l1[rep,modelNumber - 1], CSreport_l1[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                             bs_testing, clf_colorC, clf_colorS)


            BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                                BP_layer2_out, 2,                                                                                            'noskip')  # bp retrievals from layer 2
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()

            # Table 1 (memory retrievals from L2)
            print('classifiers accuracy for L2 ')
            pred_ss, pred_sc, SSreport_l2[rep,modelNumber - 1], SCreport_l2[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_l2[rep,modelNumber - 1], CSreport_l2[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                             bs_testing, clf_colorC, clf_colorS)

    SSreport_both=SSreport_both.reshape(1,-1)
    SSreport_both=SSreport_both.reshape(1,-1)
    SCreport_both=SCreport_both.reshape(1,-1)
    CCreport_both=CCreport_both.reshape(1,-1)
    CSreport_both=CSreport_both.reshape(1,-1)
    SSreport_shape=SSreport_shape.reshape(1,-1)
    SCreport_shape=SCreport_shape.reshape(1,-1)
    CCreport_shape=CCreport_shape.reshape(1,-1)
    CSreport_shape=CSreport_shape.reshape(1,-1)
    CCreport_color=CCreport_color.reshape(1,-1)
    CSreport_color=CSreport_color.reshape(1,-1)
    SSreport_color=SSreport_color.reshape(1,-1)
    SCreport_color=SCreport_color.reshape(1,-1)
    SSreport_l1=SSreport_l1.reshape(1,-1)
    SCreport_l1=SCreport_l1.reshape(1,-1)
    CCreport_l1=CCreport_l1.reshape(1,-1)
    CSreport_l1=CSreport_l1.reshape(1,-1)

    SSreport_l2= SSreport_l2.reshape(1,-1)
    SCreport_l2=SCreport_l2.reshape(1,-1)
    CCreport_l2= CCreport_l2.reshape(1,-1)
    CSreport_l2=CSreport_l2.reshape(1,-1)



    outputFile.write(
        'Table 2 both shape and color, accuracy of SS {0:.4g} SE{1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_both.mean(),
            SSreport_both.std()/math.sqrt(numModels*perms),
            SCreport_both.mean(),
            SCreport_both.std()/math.sqrt(numModels*perms),
            CCreport_both.mean(),
            CCreport_both.std()/math.sqrt(numModels*perms),
            CSreport_both.mean(),
            CSreport_both.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 shape only, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_shape.mean(),
            SSreport_shape.std()/math.sqrt(numModels*perms),
            SCreport_shape.mean(),
            SCreport_shape.std()/math.sqrt(numModels*perms),
            CCreport_shape.mean(),
            CCreport_shape.std()/math.sqrt(numModels*perms),
            CSreport_shape.mean(),
            CSreport_shape.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 color only, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g},\n accuracy of SS {4:.4g} SE {5:.4g},accuracy of SC {6:.4g} SE {7:.4g}\n'.format(
            CCreport_color.mean(),
            CCreport_color.std()/math.sqrt(numModels*perms),
            CSreport_color.mean(),
            CSreport_color.std()/math.sqrt(numModels*perms),
            SSreport_color.mean(),
            SSreport_color.std()/math.sqrt(numModels*perms),
            SCreport_color.mean(),
            SCreport_color.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 l1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l1.mean(),
            SSreport_l1.std()/math.sqrt(numModels*perms),
            SCreport_l1.mean(),
            SCreport_l1.std()/math.sqrt(numModels*perms),
            CCreport_l1.mean(),
            CCreport_l1.std()/math.sqrt(numModels*perms),
            CSreport_l1.mean(),
            CSreport_l1.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 l2, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l2.mean(),
            SSreport_l2.std()/math.sqrt(numModels*perms),
            SCreport_l2.mean(),
            SCreport_l2.std()/math.sqrt(numModels*perms),
            CCreport_l2.mean(),
            CCreport_l2.std()/math.sqrt(numModels*perms),
            CSreport_l2.mean(),
            CSreport_l2.std()/math.sqrt(numModels*perms)))



################################################ storing visual information (shape and color) along with the categorical label##########

if Tab1SuppFlag ==1:

    print('Table 1S computing the accuracy of storing labels along with shape and color information')

    ftest_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=False,transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]),download=False)
    ftest_dataset.targets= ftest_dataset.targets+10
    test_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=False,transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]),download=False)

    #build a combined dataset out of MNIST and Fasion MNIST
    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))


    perms = hugepermnum

    setSizes = [5,6,7,8]  # number of tokens

    numModels=10


    totalAccuracyShape = list()
    totalSEshape=list()
    totalAccuracyColor = list()
    totalSEcolor=list()
    totalAccuracyShape_visual=list()
    totalSEshapeVisual=list()
    totalAccuracyColor_visual=list()
    totalSEcolorVisual=list()
    totalAccuracyShapeWlabels = list()
    totalSEshapeWlabels=list()
    totalAccuracyColorWlabels = list()
    totalSEcolorWlabels=list()
    totalAccuracyShape_half=list()
    totalSEshape_half=list()
    totalAccuracyColor_half=list()
    totalSEcolor_half=list()
    totalAccuracyShape_cat=list()
    totalSEshape_cat=list()
    totalAccuracyColor_cat=list()
    totalSEcolor_cat=list()

    shape_dotplots_models=list() #data for the dot plots
    color_dotplots_models=list()
    shapeVisual_dotplots_models=list()
    colorVisual_dotplots_models=list()
    shapeWlabels_dotplots_models=list()
    colorWlabels_dotplots_models=list()
    shape_half_dotplots_models=list()
    color_half_dotplots_models=list()
    shape_cat_dotplots_models=list()
    color_cat_dotplots_models=list()


    for numItems in setSizes:
            print('Doing label/shape storage:  Setsize {num}'.format(num=numItems))
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=0)


            accuracyShapeModels=list()
            accuracyColorModels=list()
            accuracyShapeWlabelsModels=list()
            accuracyColorWlabelsModels=list()
            accuracyShapeVisualModels=list()
            accuracyColorVisualModels=list()
            accuracyShapeModels_half = list()
            accuracyColorModels_half = list()
            accuracyShapeModels_cat = list()
            accuracyColorModels_cat = list()



            for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10

                accuracyShape = list()
                accuracyColor = list()
                accuracyShape_visual=list()
                accuracyColor_visual=list()
                accuracyShape_wlabels = list()
                accuracyColor_wlabels = list()

                accuracyShape_half = list()
                accuracyColor_half = list()
                accuracyShape_cat = list()
                accuracyColor_cat = list()


                load_checkpoint(
                    'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

                print('doing model {0} for Table 1S'.format(modelNumber))
                clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
                clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
                clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
                clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


                #the ratio of visual information encoded into memory
                shape_coeff = 1
                color_coeff = 1
                shape_coeff_half=.5
                color_coeff_half=.5
                shape_coeff_cat = 0
                color_coeff_cat = 0

                for i in range(perms):
                    # print('iterstart')
                    images, shapelabels = next(iter(test_loader_smaller))  # load up a set of digits

                    imgs = images.view(-1, 3 * 28 * 28).cuda()
                    colorlabels = torch.round(
                        imgs[:, 0] * 255)

                    l1_act, l2_act, shape_act, color_act = activations(imgs)

                    shapepred, x, y, z = classifier_shapemap_test_imgs(shape_act, shapelabels, colorlabels, numItems,
                                                                       clf_shapeS, clf_shapeC)
                    colorpred, x, y, z = classifier_colormap_test_imgs(color_act, shapelabels, colorlabels, numItems,
                                                                       clf_colorC, clf_colorS)
                    # one hot coding of labels before storing into the BP
                    shape_onehot = F.one_hot(shapepred, num_classes=20)
                    shape_onehot = shape_onehot.float().cuda()
                    color_onehot = F.one_hot(colorpred, num_classes=10)
                    color_onehot = color_onehot.float().cuda()
                    #binding output when only maps are stored;  storeLabels=0
                    shape_out, color_out, L2_out, L1_out, shapelabel_junk, colorlabel_junk=BPTokens_with_labels(
                        bpsize, bpPortion, 0,shape_coeff, color_coeff, shape_act, color_act, l1_act, l2_act, shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)


                    shapepredVisual, x, ssreportVisual, z = classifier_shapemap_test_imgs(shape_out, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpredVisual, x, ccreportVisual, z = classifier_colormap_test_imgs(color_out, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)



                    #binding output that stores map activations + labels
                    shape_out_all, color_out_all, l2_out_all, l1_out_all, shape_label_out, color_label_out = BPTokens_with_labels(
                        bpsize, bpPortion, 1,shape_coeff, color_coeff, shape_act, color_act, l1_act, l2_act, shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred, x, ssreport, z = classifier_shapemap_test_imgs(shape_out_all, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred, x, ccreport, z = classifier_colormap_test_imgs(color_out_all, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel = shape_label_out.argmax(1)
                    retrievedcolorlabel = color_label_out.argmax(1)


                    # Compare accuracy against the original labels
                    accuracy_shape = torch.eq(shapelabels.cpu(), retrievedshapelabel.cpu()).sum().float() / numItems
                    accuracy_color = torch.eq(colorlabels.cpu(), retrievedcolorlabel.cpu()).sum().float() / numItems
                    accuracyShape.append(accuracy_shape)  # appends the perms
                    accuracyColor.append(accuracy_color)
                    accuracyShape_visual.append(ssreportVisual)
                    accuracyColor_visual.append(ccreportVisual)
                    accuracyShape_wlabels.append(ssreport)
                    accuracyColor_wlabels.append(ccreport)

                    # binding output that stores 50% of map activations + labels
                    shape_out_all_half, color_out_all_half, l2_out_all_half, l1_out_all_half, shape_label_out_half, color_label_out_half = BPTokens_with_labels(
                        bpsize, bpPortion, 1, shape_coeff_half, color_coeff_half, shape_act, color_act, l1_act, l2_act,
                        shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred_half, x, ssreport_half, z = classifier_shapemap_test_imgs(shape_out_all_half, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred_half, x, ccreport_half, z = classifier_colormap_test_imgs(color_out_all_half, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel_half = shape_label_out_half.argmax(1)
                    retrievedcolorlabel_half = color_label_out_half.argmax(1)

                    accuracy_shape_half = torch.eq(shapelabels.cpu(), retrievedshapelabel_half.cpu()).sum().float() / numItems
                    accuracy_color_half = torch.eq(colorlabels.cpu(), retrievedcolorlabel_half.cpu()).sum().float() / numItems
                    accuracyShape_half.append(accuracy_shape_half)  # appends the perms
                    accuracyColor_half.append(accuracy_color_half)


                      # binding output that stores only labels with 0% visual information
                    shape_out_all_cat, color_out_all_cat, l2_out_all_cat, l1_out_all_cat, shape_label_out_cat, color_label_out_cat = BPTokens_with_labels(
                        bpsize, bpPortion, 1, shape_coeff_cat, color_coeff_cat, shape_act, color_act, l1_act, l2_act,
                        shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred_cat, x, ssreport_cat, z = classifier_shapemap_test_imgs(shape_out_all_cat, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred_cat, x, ccreport_cat, z = classifier_colormap_test_imgs(color_out_all_cat, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel_cat = shape_label_out_cat.argmax(1)
                    retrievedcolorlabel_cat = color_label_out_cat.argmax(1)

                    accuracy_shape_cat = torch.eq(shapelabels.cpu(), retrievedshapelabel_cat.cpu()).sum().float() / numItems
                    accuracy_color_cat = torch.eq(colorlabels.cpu(), retrievedcolorlabel_cat.cpu()).sum().float() / numItems
                    accuracyShape_cat.append(accuracy_shape_cat)  # appends the perms
                    accuracyColor_cat.append(accuracy_color_cat)





                #append the accuracy for all models
                accuracyShapeModels.append(sum(accuracyShape) / perms)
                accuracyColorModels.append(sum(accuracyColor) / perms)
                accuracyShapeVisualModels.append(sum(accuracyShape_visual)/perms)
                accuracyColorVisualModels.append(sum(accuracyColor_visual)/perms)
                accuracyShapeWlabelsModels.append(sum(accuracyShape_wlabels) / perms)
                accuracyColorWlabelsModels.append(sum(accuracyColor_wlabels) / perms)
                accuracyShapeModels_half.append(sum(accuracyShape_half) / perms)
                accuracyColorModels_half.append(sum(accuracyColor_half) / perms)
                accuracyShapeModels_cat.append(sum(accuracyShape_cat) / perms)
                accuracyColorModels_cat.append(sum(accuracyColor_cat) / perms)


            shape_dotplots_models.append(torch.stack(accuracyShapeModels).view(1,-1))
            totalAccuracyShape.append(torch.stack(accuracyShapeModels).mean())
            totalSEshape.append(torch.stack(accuracyShapeModels).std()/math.sqrt(numModels))

            color_dotplots_models.append(torch.stack(accuracyColorModels).view(1,-1))
            totalAccuracyColor.append(torch.stack(accuracyColorModels).mean())
            totalSEcolor.append(torch.stack(accuracyColorModels).std() / math.sqrt(numModels))

            shapeVisual_dotplots_models.append(torch.stack(accuracyShapeVisualModels).view(1,-1))
            totalAccuracyShape_visual.append(torch.stack(accuracyShapeVisualModels).mean())
            totalSEshapeVisual.append(torch.stack(accuracyShapeVisualModels).std()/math.sqrt(numModels))

            colorVisual_dotplots_models.append(torch.stack(accuracyColorVisualModels).view(1,-1))
            totalAccuracyColor_visual.append(torch.stack(accuracyColorVisualModels).mean())
            totalSEcolorVisual.append(torch.stack(accuracyColorVisualModels).std() / math.sqrt(numModels))

            shapeWlabels_dotplots_models.append(torch.stack(accuracyShapeWlabelsModels).view(1,-1))
            totalAccuracyShapeWlabels .append(torch.stack(accuracyShapeWlabelsModels).mean())
            totalSEshapeWlabels.append(torch.stack(accuracyShapeWlabelsModels).std() / math.sqrt(numModels))

            colorWlabels_dotplots_models.append(torch.stack(accuracyColorWlabelsModels).view(1,-1))
            totalAccuracyColorWlabels.append(torch.stack(accuracyColorWlabelsModels).mean())
            totalSEcolorWlabels.append(torch.stack(accuracyColorWlabelsModels).std() / math.sqrt(numModels))

            shape_half_dotplots_models.append(torch.stack(accuracyShapeModels_half).view(1,-1))
            totalAccuracyShape_half.append(torch.stack(accuracyShapeModels_half).mean())
            totalSEshape_half.append(torch.stack(accuracyShapeModels_half).std() / math.sqrt(numModels))

            color_half_dotplots_models.append(torch.stack(accuracyColorModels_half).view(1,-1))
            totalAccuracyColor_half.append(torch.stack(accuracyColorModels_half).mean())
            totalSEcolor_half.append(torch.stack(accuracyColorModels_half).std() / math.sqrt(numModels))

            shape_cat_dotplots_models.append(torch.stack(accuracyShapeModels_cat).view(1,-1))
            totalAccuracyShape_cat.append(torch.stack(accuracyShapeModels_cat).mean())
            totalSEshape_cat.append(torch.stack(accuracyShapeModels_cat).std() / math.sqrt(numModels))

            color_cat_dotplots_models.append(torch.stack(accuracyColorModels_cat).view(1,-1))
            totalAccuracyColor_cat.append(torch.stack(accuracyColorModels_cat).mean())
            totalSEcolor_cat.append(torch.stack(accuracyColorModels_cat).std() / math.sqrt(numModels))







    print(shape_dotplots_models)
    print(color_dotplots_models)
    print(shapeVisual_dotplots_models)
    print(colorVisual_dotplots_models)
    print(shapeWlabels_dotplots_models)
    print(colorWlabels_dotplots_models)
    print(shape_half_dotplots_models)
    print(color_half_dotplots_models)
    print(shape_cat_dotplots_models)
    print(color_cat_dotplots_models)

    outputFile.write('Table 1S, accuracy of ShapeLabel')


    for i in range(len(setSizes)):
        outputFile.write('\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape[i],totalSEshape[i] ))

    outputFile.write('\n\nTable 3, accuracy of ColorLabel')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor[i], totalSEcolor[i]))

    outputFile.write('\n\nTable 3, accuracy of shape map with no labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_visual[i], totalSEshapeVisual[i]))

    outputFile.write('\n\nTable 3, accuracy of color map with no labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_visual[i],
                                                                 totalSEcolorVisual[i]))

    outputFile.write('\n\nTable 3, accuracy of shape map with labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShapeWlabels[i],
                                                                totalSEshapeWlabels[i]))

    outputFile.write('\n\nTable 3, accuracy of color map with labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColorWlabels[i],
                                                                 totalSEcolorWlabels[i]))

    outputFile.write('\n\nTable 3, accuracy of ShapeLabel for 50% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_half[i], totalSEshape_half[i]))

    outputFile.write('\n\nTable 3, accuracy of ColorLabel for 50% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_half[i], totalSEcolor_half[i]))

    outputFile.write('\n\nTable 3, accuracy of ShapeLabel for 0% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_cat[i], totalSEshape_cat[i]))

    outputFile.write('\n\nTable 3, accuracy of ColorLabel for 0% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_cat[i], totalSEcolor_cat[i]))

if TabTwoColorFlag1 == 1:
    modelNumber=1

    print('doing model {0} for Table 1'.format(modelNumber))
    clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
    clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
    clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
    clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


    print('Doing Table 1')
    numimg = 10
    trans2 = transforms.ToTensor()
    imgs_all = []
    convert_tensor = transforms.ToTensor()
    convert_image = transforms.ToPILImage()
    #for i in range (1,numimg+1):
    img = Image.open('{each}_thick.png'.format(each=9))
    img = convert_tensor(img)
    img_new = img[0:3,:,:]*1  #Ian

    imgs_all.append(img_new)
    imgs_all = torch.stack(imgs_all)
    imgs = imgs_all.view(-1, 3 * 28 * 28).cuda()
    img_new = convert_image(img_new)
    save_image(
            torch.cat([trans2(img_new).view(1, 3, 28, 28)], 0),
            'output{num}/figure10test.png'.format(num=modelNumber),
            nrow=numimg,
            normalize=False,
            range=(-1, 1),
        )

            # run them all through the encoder
    l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

    # now store and retrieve them from the BP
    BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                 shape_act, color_act,
                                                                                 shape_coeff, color_coeff, 1, 1,
                                                                                 normalize_fact_familiar)
    BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act,
                                                                                           shape_coeff, 0, 0, 0,
                                                                                           normalize_fact_familiar)
    BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act, 0,
                                                                                           color_coeff, 0, 0,
                                                                                           normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0,
                                                                                l1_coeff, 0,
                                                                                normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0, 0,
                                                                                l2_coeff,
                                                                                normalize_fact_familiar)
    #print(clf_colorC, clf_colorS)
    pred_cc,pred_cs,pred_cb = classifier_color_prediction(clf_colorC,color_out_BP_coloronly, color_out_BP_shapeonly,color_out_BP_both)
    #outputFile.write(f'Color from color:{pred_cc}, Color from shape:{pred_cs}')# Color from both:{pred_cb}')
    print(pred_cc)
if TabTwoColorFlag == 1:
    numModels=1
    perms=1
    SSreport_both = np.tile(0.0, [perms,numModels])
    SCreport_both = np.tile(0.0, [perms,numModels])
    CCreport_both = np.tile(0.0, [perms,numModels])
    CSreport_both = np.tile(0.0, [perms,numModels])
    SSreport_shape = np.tile(0.0, [perms,numModels])
    SCreport_shape = np.tile(0.0, [perms,numModels])
    CCreport_shape = np.tile(0.0, [perms,numModels])
    CSreport_shape = np.tile(0.0, [perms,numModels])
    SSreport_color = np.tile(0.0, [perms,numModels])
    SCreport_color = np.tile(0.0, [perms,numModels])
    CCreport_color = np.tile(0.0, [perms,numModels])
    CSreport_color = np.tile(0.0, [perms,numModels])
    SSreport_l1 = np.tile(0.0, [perms,numModels])
    SCreport_l1= np.tile(0.0, [perms,numModels])
    CCreport_l1 = np.tile(0.0, [perms,numModels])
    CSreport_l1 = np.tile(0.0, [perms,numModels])
    SSreport_l2 = np.tile(0.0, [perms,numModels])
    SCreport_l2 = np.tile(0.0, [perms,numModels])
    CCreport_l2 = np.tile(0.0, [perms,numModels])
    CSreport_l2 = np.tile(0.0, [perms,numModels])


    #for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
    #load_checkpoint(
            #'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))
    modelNumber=1

    print('doing model {0} for Table 1'.format(modelNumber))
    clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
    clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
    clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
    clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


    print('Doing Table 1')
    numimg = 10
    imgs_all = []
    trans2 = transforms.ToTensor()

    convert_tensor = transforms.ToTensor()
    convert_image = transforms.ToPILImage()
    for i in range (1,numimg+1):
        img = Image.open('{each}_thick.png'.format(each=i))
        img = convert_tensor(img)
        img_new = img[0:3,:,:]*1   #Ian

        imgs_all.append(img_new)
    imgs_all = torch.stack(imgs_all)
    imgs = imgs_all.view(-1, 3 * 28 * 28).cuda()
    img_new = convert_image(img_new)
    save_image(
            torch.cat([trans2(img_new).view(1, 3, 28, 28)], 0),
            'output{num}/figure10test.png'.format(num=modelNumber),
            nrow=numimg,
            normalize=False,
            range=(-1, 1),
        )
    #slice labels to prevent an error
    colorlabels = colorlabels[0:numimg]
    shapelabels = shapelabels[0:numimg]
    print(colorlabels)
    print(shapelabels)
            # run them all through the encoder
    l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

    # now store and retrieve them from the BP
    BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                 shape_act, color_act,
                                                                                 shape_coeff, color_coeff, 1, 1,
                                                                                 normalize_fact_familiar)
    BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act,
                                                                                           shape_coeff, 0, 0, 0,
                                                                                           normalize_fact_familiar)
    BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act, 0,
                                                                                           color_coeff, 0, 0,
                                                                                           normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0,
                                                                                l1_coeff, 0,
                                                                                normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0, 0,
                                                                                l2_coeff,
                                                                                normalize_fact_familiar)

    # Table 1: classifier accuracy for shape and color for memory retrievals
    print('classifiers accuracy for memory retrievals of BN_both for Table 1')
    pred_ss, pred_sc, SSreport_both[rep,modelNumber-1], SCreport_both[rep,modelNumber-1] = classifier_shapemap_test_imgs(shape_out_BP_both, shapelabels,
                                                                     colorlabels, bs_testing, clf_shapeS,
                                                                     clf_shapeC)
    pred_cc, pred_cs, CCreport_both[rep,modelNumber-1], CSreport_both[rep,modelNumber-1]= classifier_colormap_test_imgs(color_out_BP_both, shapelabels,
                                                                     colorlabels, bs_testing, clf_colorC,clf_colorS)



    # Table 1: classifier accuracy for shape and color for memory retrievals
    print('classifiers accuracy for memory retrievals of BN_shapeonly for Table 1')
    pred_ss, pred_sc, SSreport_shape[rep,modelNumber - 1], SCreport_shape[
    rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_shapeonly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_shape[rep,modelNumber - 1], CSreport_shape[
    rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_shapeonly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_colorC, clf_colorS)

    print('classifiers accuracy for memory retrievals of BN_coloronly for Table 1')
    pred_ss, pred_sc, SSreport_color[rep,modelNumber - 1], SCreport_color[
    rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_coloronly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_color[rep,modelNumber - 1], CSreport_color[
    rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_coloronly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_colorC, clf_colorS)


    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                        BP_layer2_out, 1,
                                                                                        'noskip')  # bp retrievals from layer 1
    z_color = vae.sampling(mu_color, log_var_color).cuda()
    z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
    # Table 1 (memory retrievals from L1)
    print('classifiers accuracy for L1 ')
    
    pred_ss, pred_sc, SSreport_l1[rep,modelNumber - 1], SCreport_l1[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_l1[rep,modelNumber - 1], CSreport_l1[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                     bs_testing, clf_colorC, clf_colorS)


    BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                        BP_layer2_out, 2,                                                                                            'noskip')  # bp retrievals from layer 2
    z_color = vae.sampling(mu_color, log_var_color).cuda()
    z_shape = vae.sampling(mu_shape, log_var_shape).cuda()

    # Table 1 (memory retrievals from L2)
    print('classifiers accuracy for L2 ')
    pred_ss, pred_sc, SSreport_l2[rep,modelNumber - 1], SCreport_l2[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_l2[rep,modelNumber - 1], CSreport_l2[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                     bs_testing, clf_colorC, clf_colorS)
                                                                     
    SSreport_both=SSreport_both.reshape(1,-1)
    SSreport_both=SSreport_both.reshape(1,-1)
    SCreport_both=SCreport_both.reshape(1,-1)
    CCreport_both=CCreport_both.reshape(1,-1)
    CSreport_both=CSreport_both.reshape(1,-1)
    SSreport_shape=SSreport_shape.reshape(1,-1)
    SCreport_shape=SCreport_shape.reshape(1,-1)
    CCreport_shape=CCreport_shape.reshape(1,-1)
    CSreport_shape=CSreport_shape.reshape(1,-1)
    CCreport_color=CCreport_color.reshape(1,-1)
    CSreport_color=CSreport_color.reshape(1,-1)
    SSreport_color=SSreport_color.reshape(1,-1)
    SCreport_color=SCreport_color.reshape(1,-1)
    SSreport_l1=SSreport_l1.reshape(1,-1)
    SCreport_l1=SCreport_l1.reshape(1,-1)
    CCreport_l1=CCreport_l1.reshape(1,-1)
    CSreport_l1=CSreport_l1.reshape(1,-1)

    SSreport_l2= SSreport_l2.reshape(1,-1)
    SCreport_l2=SCreport_l2.reshape(1,-1)
    CCreport_l2= CCreport_l2.reshape(1,-1)
    CSreport_l2=CSreport_l2.reshape(1,-1)



    outputFile.write(
        'TableTwoColor both shape and color, accuracy of SS {0:.4g} SE{1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_both.mean(),
            SSreport_both.std()/math.sqrt(numModels*perms),
            SCreport_both.mean(),
            SCreport_both.std()/math.sqrt(numModels*perms),
            CCreport_both.mean(),
            CCreport_both.std()/math.sqrt(numModels*perms),
            CSreport_both.mean(),
            CSreport_both.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor shape only, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_shape.mean(),
            SSreport_shape.std()/math.sqrt(numModels*perms),
            SCreport_shape.mean(),
            SCreport_shape.std()/math.sqrt(numModels*perms),
            CCreport_shape.mean(),
            CCreport_shape.std()/math.sqrt(numModels*perms),
            CSreport_shape.mean(),
            CSreport_shape.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor color only, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g},\n accuracy of SS {4:.4g} SE {5:.4g},accuracy of SC {6:.4g} SE {7:.4g}\n'.format(
            CCreport_color.mean(),
            CCreport_color.std()/math.sqrt(numModels*perms),
            CSreport_color.mean(),
            CSreport_color.std()/math.sqrt(numModels*perms),
            SSreport_color.mean(),
            SSreport_color.std()/math.sqrt(numModels*perms),
            SCreport_color.mean(),
            SCreport_color.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor l1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l1.mean(),
            SSreport_l1.std()/math.sqrt(numModels*perms),
            SCreport_l1.mean(),
            SCreport_l1.std()/math.sqrt(numModels*perms),
            CCreport_l1.mean(),
            CCreport_l1.std()/math.sqrt(numModels*perms),
            CSreport_l1.mean(),
            CSreport_l1.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor l2, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l2.mean(),
            SSreport_l2.std()/math.sqrt(numModels*perms),
            SCreport_l2.mean(),
            SCreport_l2.std()/math.sqrt(numModels*perms),
            CCreport_l2.mean(),
            CCreport_l2.std()/math.sqrt(numModels*perms),
            CSreport_l2.mean(),
            CSreport_l2.std()/math.sqrt(numModels*perms)))


######This part is to detect whether a stimulus is novel or familiar

if noveltyDetectionFlag==1:


    perms=smallpermnum
    numModels=10

    acc_fam=torch.zeros(numModels,perms)
    acc_nov=torch.zeros(numModels,perms)

    for modelNumber in range (1,numModels+1):

        acc_fam[modelNumber-1,:], acc_nov[modelNumber-1,:]= novelty_detect( perms, bpsize, bpPortion, shape_coeff, color_coeff, normalize_fact_familiar,
                  normalize_fact_novel, modelNumber, test_loader_smaller)
    mean_fam=acc_fam.view(1,-1).mean()
    fam_SE= acc_fam.view(1,-1).std()/(len(acc_fam.view(1,-1)))

    mean_nov=acc_nov.view(1,-1).mean()
    nov_SE= acc_nov.view(1,-1).std()/(len(acc_nov.view(1,-1)))





    outputFile.write(
            '\accuracy of detecting the familiar shapes : mean is {0:.4g} and SE is {1:.4g} '.format(mean_fam, fam_SE))

    outputFile.write(
            '\naccuracy of detecting the novel shapes : mean is {0:.4g} and SE is {1:.4g} '.format(mean_nov, nov_SE))

#outputFile.close()

def plotbpvals(set1,set2,set3,set4,set5,label):

    #plot the values of the BP nodes..  this should be made into a function
    plt.figure()
    plt.subplot(151)
    plt.hist(set1, 100)
    plt.xlim(-10, 10)
    plt.subplot(152)
    plt.hist(set2, 100)
    plt.xlim(-10,10)
    plt.subplot(153)
    plt.hist(set3, 100)
    plt.subplot(154)
    plt.hist(set4, 100)
    plt.ylabel(label)
    plt.subplot(155)
    plt.hist(set5, 100)
    plt.show()     #visualize the values of the BP as a distribution
