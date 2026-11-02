import numpy as np
from MLR_src.mVAE import load_checkpoint, vae_builder, load_dimensions, VAE_CNN
from sklearn.cluster import KMeans
import torch
from PIL import Image, ImageOps
from collections import defaultdict
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms as torch_transforms
import joblib
import os
import math
from sklearn.mixture import GaussianMixture

DATASET_ROOT = '/home/bwyble/data/'

def preprocess_quickdraw(base_dataset, model_name):
    if not os.path.exists(f'data/preprocessed_quickdraw_{model_name}.pkl') or not os.path.exists(f'data/preprocessed_quickdraw_indices_{model_name}.pkl'):
        dataset_dict = defaultdict(list)
        index_dict = defaultdict(list) # index mapping between basedataset and dataset_dict
        print('classes:', np.unique(base_dataset[:,-1]))
        for i in range(len(base_dataset)):
            image = base_dataset[i, :-1].reshape(28, 28)  # image
            np_img = np.dstack([image, image, image])
            img = Image.fromarray(np_img, 'RGB')
            target = int(base_dataset[i, -1])  # label
            dataset_dict[target] += [torch_transforms.ToTensor()(img).view(1,3,28,28)]
            index_dict[target] += [i]

        #save_image(dataset_dict[0][1], 'sample123.png', pad_value=0.6)
        joblib.dump(dataset_dict, f'data/preprocessed_quickdraw_{model_name}.pkl')
        joblib.dump(index_dict, f'data/preprocessed_quickdraw_indices_{model_name}.pkl')
        print('data preprocessing done')
        return dataset_dict, index_dict
    
    else:
        print('data loading')
        dataset_dict = joblib.load(f'data/preprocessed_quickdraw_{model_name}.pkl')
        index_dict = joblib.load(f'data/preprocessed_quickdraw_indices_{model_name}.pkl')

    return dataset_dict, index_dict

@torch.no_grad()
def filter_quickdraw(model, base_dataset, n_clusters=10, d=1, model_name='123'):
    print('preprocessing_quickdraw')
    index_list = [0, 2, 7, 8, 10, 11] # [0,1,2,3,4,5,6,7,8,9,10,11] #  
    
    index_built = [os.path.exists(f'data/object_act_class_{model_name}_{i}.pkl') for i in index_list]
    if not all(index_built):
        # run if switching dataset or loading a different VAE version:
        data_dict, index_dict = preprocess_quickdraw(base_dataset, 'VAE_CNN')
    
    else:
        # run if preprocessed data for the same dataset and VAE version already exists:
        index_dict = joblib.load(f'data/preprocessed_quickdraw_indices_VAE_CNN.pkl')
    #kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    results = {}
    # which classes to keep for the filtered set:
    
    for i in index_list: #index_dict.keys():
        print('i:', i)

        if not os.path.exists(f'data/object_act_class_{model_name}_{i}.pkl'):
            # memory management
            object_act = []
            for j in range(1, len(data_dict[i])):
                # convert sample list to tensor
                samples = torch.stack(data_dict[i][(j-1):j], dim=0).view(-1,3,28,28).to(d)  # [N, 28, 28]
                #save_image(samples[:5], 'sample1234.png', pad_value=0.6)
                print(type(samples), samples.size())
                activations = model.activations(samples)
                t_object_act = activations['object'].to('cpu')  # [1000, 12]
                object_act += [t_object_act]
            
            object_act = torch.cat(object_act, dim=0)  # [N, 12]
            joblib.dump(object_act, f'data/object_act_class_{model_name}_{i}.pkl')
        else:
            object_act = joblib.load(f'data/object_act_class_{model_name}_{i}.pkl')
        
        print('object_act:', object_act.size())
        
        
        '''labels = kmeans.fit_predict(object_act)
        cluster_sizes = np.bincount(labels)
        max_cluster = np.argmax(cluster_sizes)
        print('max_cluster:', max_cluster)

        centroid = kmeans.cluster_centers_[[max_cluster]]

        dists = np.linalg.norm(object_act - centroid, axis=1)    # [70000]
        selected_indices = np.argsort(dists)[:200]'''
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(object_act)

        cluster_sizes = np.bincount(labels)
        sample_count = 40
        # use selected clock index and find closest samples to it
        if i == 100: #clock
            # choose the 400 samples closest to clock sample number 87
            clock_sample_idx = 87
            clock_sample = object_act[clock_sample_idx]
            dists = np.linalg.norm(object_act - clock_sample, axis=1)
            selected_indices = np.argsort(dists)[:sample_count]
            '''elif i == 8: #sailboat
            # choose the 400 samples closest to sailboat sample number 226
            sailboat_sample_idx = 990
            sailboat_sample = object_act[sailboat_sample_idx]
            dists = np.linalg.norm(object_act - sailboat_sample, axis=1)
            selected_indices = np.argsort(dists)[:sample_count]'''
        else:
            max_cluster = np.argmax(cluster_sizes)
            print('max_cluster:', max_cluster)

            probs = gmm.predict_proba(object_act)[:, max_cluster]
            selected_indices = np.argsort(probs)[::-1][:sample_count]
        
        #print(selected_indices)
        result_indices = [index_dict[i][idx] for idx in selected_indices]

        results[i] = result_indices

    return results


def save_filtered_images(base_dataset, filtered_indices):
    grid_cols = len(filtered_indices[next(iter(filtered_indices))])//10
    print('filtered_indices keys:', filtered_indices.keys())
    os.makedirs('filtered_images', exist_ok=True)
    filtered_dataset = []
    for class_id, indices in filtered_indices.items():
        images = []
        for idx in indices:
            filtered_dataset.append(base_dataset[idx])
            image = base_dataset[idx, :-1].reshape(28, 28)
            np_img = np.dstack([image, image, image])
            images.append(np_img)

        # Build grid
        n = len(images)
        grid_rows = math.ceil(n / grid_cols)
        grid = np.zeros((grid_rows * 28, grid_cols * 28, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row, col = divmod(i, grid_cols)
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = img

        Image.fromarray(grid, 'RGB').save(f'filtered_images/class_{class_id}_grid.png')
        print(f'Saved grid for class {class_id}: {grid_rows}x{grid_cols} ({n} images)')
    filtered_dataset = np.array(filtered_dataset)
    np.save(f'{DATASET_ROOT}quickdraw_npy/filtered_dataset_label_net.npy', filtered_dataset)

base_dataset = np.load(f'{DATASET_ROOT}quickdraw_npy/filtered_dataset_1.npy')
print(base_dataset.shape)
folder_name = "bg-col-6000"
checkpoint_folder_path = f'checkpoints/{folder_name}'
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d=1, draw=True)
vae.eval()

filtered_indices = filter_quickdraw(vae, base_dataset, n_clusters=40, d= 1, model_name=folder_name+'2')
save_filtered_images(base_dataset, filtered_indices)

