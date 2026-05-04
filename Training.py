import sys
import time
import argparse

parser = argparse.ArgumentParser(description="Training of MLR-2.0")
parser.add_argument("--load_prev", type=bool, default=False, help="Begin training from previous checkpoints")
parser.add_argument("--cuda", type=bool, default=True, help="Cuda availability")
parser.add_argument("--cuda_device", type=int, default=1, help="Which cuda device to use")
parser.add_argument("--folder", type=str, default='test', help="Where to store checkpoints in checkpoints/")
# VVV defines which components are trained
parser.add_argument("--components", nargs='+', type=str, default=['shape', 'color', 'retinal', 'object', 'skip_cropped', 'cropped', 'retinal_object', 'cropped_object'], help="Which components to train")
#parser.add_argument("--components", nargs='+', type=str, default=['shape', 'color', 'retinal', 'skip_cropped', 'cropped'], help="Which components to train")
parser.add_argument("--z_dim", type=int, default=12, help="Size of the mVAE latent dimension")
parser.add_argument("--train_list", nargs='+', type=str, default=['mVAE', 'label_net', 'SVM'], help="Which models to train")
parser.add_argument("--wandb", type=bool, default=False, help="Track training with wandb")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--start_ep", type=int, default=1, help="what epoch to resume training")
parser.add_argument("--end_ep", type=int, default=100, help="what epoch to train to")
#parser.add_argument("--batch_size", nargs='+', type=int, default=['mVAE', 'label_net', 'SVM'], help="Which components to train")
args = parser.parse_args()


# prerequisites
import torch
import os
from MLR_src.mVAE import load_checkpoint, vae_builder, load_dimensions
#from torch.utils.data import DataLoader, ConcatDataset
from MLR_src.dataset_builder import Dataset
from MLR_src.train_mVAE import train_mVAE
from MLR_src.label_network import train_labelnet
from MLR_src.train_classifiers_revised import train_classifiers
from training_constants import training_datasets, training_components
from itertools import cycle
from torchvision.utils import save_image

folder_name = args.folder
checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions

if not os.path.exists('training_samples/'):
    os.mkdir('training_samples/')

if not os.path.exists('checkpoints/'):
    os.mkdir('checkpoints/')

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

if not os.path.exists(f'training_samples/{folder_name}/'):
    os.mkdir(f'training_samples/{folder_name}/')

if args.cuda is True:
    d = args.cuda_device   #which cuda GPU?

#use a previous checkpoint?
if 'mVAE' not in args.train_list:  # we HAVE to load because no MVAE was trained
    load = True
else:
    load = args.load_prev  



print(f'Device: {d}')
print(f'Load: {load}')

if torch.cuda.is_available():
    device = torch.device(f'cuda:{d}')
    torch.cuda.set_device(d)
    print('CUDA')
else:
    device = 'cpu'
    print('CUDA not available')

bs=100   #batch size for training the main VAE
SVM_bs = 1000  #batch size for training the svm classifiers
obj_latent_flag = True   #this flag determines whether the VAE has an obj latent space


if load is True:
    vae = load_checkpoint(f'{checkpoint_folder_path}{args.checkpoint_name}', d, obj_latent_flag)
    dimensions = load_dimensions(f'{checkpoint_folder_path}/{args.checkpoint_name}', d)
    print('checkpoint loaded from folder'+checkpoint_folder_path +args.checkpoint_name)     

else:
    dimensions = [-1, -1, 128, args.z_dim]
    vae, dimensions = vae_builder(dimensions, obj_latent_flag)

#vae = nn.DataParallel(vae)

dataloaders = {}
SVM_dataloaders = {}
weighted_components = [] #specifies the order/frequency the model latents will be trained


# init dataloaders for mVAE and label training by making sure the data sets for each model component are added
# model components are the latent spaces, like shape, color, etc   Each component also has a specific list of transforms

for component in args.components:
    weight = training_components[component][1]
    weighted_components += [component] * weight
    for dataset in training_components[component][0]:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]  #load the transforms for this dataset
        dataloader = cycle(Dataset(dataset_name, dataset_transforms).get_loader(bs//len(training_components[component][0])))
        dataloaders[dataset] = iter(dataloader)

# init dataloaders for SVM training
for component in args.components:
    for dataset in training_components[component][0]:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]   #load the transforms for this dataset
        SVM_dataloaders[dataset] = cycle(Dataset(dataset_name, dataset_transforms).get_loader(SVM_bs))


vae.to(device)

#print(dataloaders.keys())

print(f'Training: {args.train_list}')
epoch_count = args.end_ep

data, labels = next(iter(dataloaders['emnist-map']))
image = data[1].cuda()

with torch.no_grad():
    recon, _, _, _, _,_,_ = vae(image, 'shape', ['shape'])

save_image(
    torch.cat([image[:25], recon[:25]], 0),
    'quick_test_recon.png', nrow=25)



#train mVAE
if 'mVAE' in args.train_list:
    print('Training: mVAE')
    train_mVAE(dataloaders, weighted_components, vae, epoch_count, folder_name, args.wandb, args.start_ep, dimensions)

#train_labels
if 'label_net' in args.train_list:
    print('Training: label networks')
    train_labelnet(dataloaders, vae, 15, args.z_dim, folder_name, args.components)

#train_classifiers
if 'SVM' in args.train_list:
    print('Training: classifiers')
    train_classifiers(SVM_dataloaders, vae, folder_name)
