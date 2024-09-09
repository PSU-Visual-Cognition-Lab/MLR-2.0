import sys

if len(sys.argv[1:]) != 0:
    d = int(sys.argv[1:][0])
    load = False#bool(sys.argv[1:][1])
else:
    d=1
    load = False

# prerequisites
import torch
import os
from MLR_src.mVAE import load_checkpoint, vae_builder
from torch.utils.data import DataLoader, ConcatDataset
from MLR_src.dataset_builder import Dataset
from MLR_src.train_mVAE import train_mVAE

folder_name = 'test'
#torch.set_default_dtype(torch.float64)
checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

if len(sys.argv[1:]) != 0:
    d = int(sys.argv[1:][0])
else:
    d=1
print(f'Device: {d}')
print(f'Load: {load}')

if torch.cuda.is_available():
    device = torch.device(f'cuda:{d}')
    torch.cuda.set_device(d)
    print('CUDA')
else:
    device = 'cpu'

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
if load is True:
    load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d)
    print('checkpoint loaded')

bs=100
vae, z_dim = vae_builder()

'''# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
#emnist_transforms = {'retina':True, 'colorize':True}
mnist_transforms = {'retina':True, 'colorize':True, 'scale':False, 'build_retina':False}
mnist_test_transforms = {'retina':True, 'colorize':True, 'scale':False}
skip_transforms = {'skip':True, 'colorize':True}

#emnist_dataset = Dataset('emnist', emnist_transforms)
mnist_dataset = Dataset('mnist', mnist_transforms)

#emnist_test_dataset = Dataset('emnist', emnist_transforms, train= False)
mnist_test_dataset = Dataset('mnist', mnist_test_transforms, train= False)

#blocks
#block_dataset = torch.load('original_1.pth').cuda()

#emnist_skip = Dataset('emnist', skip_transforms)
mnist_skip = Dataset('mnist', skip_transforms)

#concat datasets and init dataloaders
train_loader_noSkip = mnist_dataset.get_loader(bs)
#sample_loader_noSkip = mnist_dataset.get_loader(25)
test_loader_noSkip = mnist_test_dataset.get_loader(bs)
#mnist_skip = torch.utils.data.DataLoader(dataset=ConcatDataset([block_dataset, mnist_skip]), batch_size=bs, shuffle=True,  drop_last= True)
mnist_skip = mnist_skip.get_loader(bs)'''

#add colorsquares dataset to training
vae.to(device)

dataloaders = []#[train_loader_noSkip, None, mnist_skip, test_loader_noSkip, None]

train_mVAE(dataloaders, vae, 1000, folder_name, False)

#train_labels

#train_classifiers