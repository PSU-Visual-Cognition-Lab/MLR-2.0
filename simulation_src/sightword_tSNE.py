from mVAE import vae, image_activations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset_builder import dataset_builder
from torch.utils.data import DataLoader, ConcatDataset
from mpl_toolkits.mplot3d import Axes3D

vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

device ='cuda'
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    return vae
modelNumber = 1
v = '' #'_v1'
cur_dataset = 'emnist'
'''load_checkpoint(f'output_emnist_recurr{v}/checkpoint_300.pth') # MLR2.0 trained on emnist letters, digits, and fashion mnist

vae.eval()

bs = 1
# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
emnist_dataset, emnist_skip, emnist_test_dataset = dataset_builder('emnist', bs, None, False, None, False, True)
mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', bs, None, False, None, False, True)
#fmnist_dataset, fmnist_skip, fmnist_test_dataset = dataset_builder('fashion_mnist', bs, None, False, None, False, True)

#concat datasets and init dataloaders
train_loader = torch.utils.data.DataLoader(dataset=ConcatDataset([mnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)


torch.save([], f'{cur_dataset}rep.pt')

reps = []
c = 0
for data_tup in tqdm(train_loader):
    z_shape, z_color, z_location = image_activations(data_tup[0].view(-1,3,28,28).cuda())
    reps += [(z_shape,data_tup[1][0].item())]
    #print(data_tup[1][0].item())
    if len(reps)==1200:
        x = torch.load(f'{cur_dataset}rep.pt')
        out = x + reps
        torch.save(out, f'{cur_dataset}rep.pt')
        torch.cuda.empty_cache()
        reps=[]
    if c > 6000:
        break
    c += 1

print('Samples done')

torch.cuda.empty_cache()'''

data= torch.load(f'{cur_dataset}rep.pt')

data = data [:3000]

tensors = [item[0][0].cpu().detach().numpy() for item in data]
tensors_array = np.array(tensors)

tsne = TSNE(n_components=2, perplexity=30)
embedded_data = tsne.fit_transform(tensors_array)

labels = [item[1] for item in data]
unique_classes = list(set(labels))
class_to_color = {cls: plt.cm.jet(i / len(unique_classes)) for i, cls in enumerate(unique_classes)}

plt.figure()
for i, cls in enumerate(unique_classes):
    class_indices = [j for j, label in enumerate(labels) if label == cls]
    plt.scatter(embedded_data[class_indices, 0], embedded_data[class_indices, 1],
                label=f'Class {vals[cls]}', color=class_to_color[cls], marker='o')

    repr_x_coord = sum(embedded_data[class_indices, 0]) / (len(embedded_data[class_indices, 0]))
    repr_y_coord = sum(embedded_data[class_indices, 1]) / (len(embedded_data[class_indices, 1]))
    representative_index = class_indices[random.randint(min(unique_classes), max(unique_classes) + 1)] #
    plt.annotate(f'{vals[cls]}', (embedded_data[representative_index, 0], embedded_data[representative_index, 1]), fontsize=20)

plt.legend()
plt.show()