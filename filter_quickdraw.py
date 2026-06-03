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

def preprocess_quickdraw(base_dataset):
    if not os.path.exists('data/preprocessed_quickdraw.pkl') or not os.path.exists('data/preprocessed_quickdraw_indices.pkl'):
        dataset_dict = defaultdict(list)
        index_dict = defaultdict(list) # index mapping between basedataset and dataset_dict
        for i in range(len(base_dataset)):
            image = base_dataset[i, :-1].reshape(28, 28)  # image
            np_img = np.dstack([image, image, image])
            img = Image.fromarray(np_img, 'RGB')
            target = int(base_dataset[i, -1])  # label
            dataset_dict[target] += [torch_transforms.ToTensor()(img).view(1,3,28,28)]
            index_dict[target] += [i]

        #save_image(dataset_dict[0][1], 'sample123.png', pad_value=0.6)
        joblib.dump(dataset_dict, 'data/preprocessed_quickdraw.pkl')
        joblib.dump(index_dict, 'data/preprocessed_quickdraw_indices.pkl')
        print('data preprocessing done')
        return dataset_dict, index_dict
    
    else:
        print('data loading')
        dataset_dict = joblib.load('data/preprocessed_quickdraw.pkl')
        index_dict = joblib.load('data/preprocessed_quickdraw_indices.pkl')
    
    return dataset_dict, index_dict

@torch.no_grad()
def filter_quickdraw(model, base_dataset, n_clusters=10, d=1):
    print('preprocessing_quickdraw')
    # run if switching dataset or loading a different VAE version:
    #data_dict, index_dict = preprocess_quickdraw(base_dataset)
    # run if preprocessed data for the same dataset and VAE version already exists:
    index_dict = joblib.load('data/preprocessed_quickdraw_indices.pkl')
    #kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    results = {}
    # which classes to keep for the filtered set:
    index_list = [0,2,7,8,9]

    for i in index_list: #index_dict.keys():
        print('i:', i)

        if not os.path.exists(f'data/object_act_class_{i}.pkl'):
            # memory management
            object_act = []
            for j in range(1, len(data_dict[i])//1000 + 1):
                # convert sample list to tensor
                samples = torch.stack(data_dict[i][(j-1)*1000:j*1000], dim=0).view(-1,3,28,28).to(d)  # [N, 28, 28]
                #save_image(samples[:5], 'sample1234.png', pad_value=0.6)
                print(type(samples), samples.size())
                activations = model.activations(samples)
                t_object_act = activations['object'].to('cpu')  # [1000, 12]
                object_act += [t_object_act]
            
            object_act = torch.cat(object_act, dim=0)  # [N, 12]
            joblib.dump(object_act, f'data/object_act_class_{i}.pkl')
        else:
            object_act = joblib.load(f'data/object_act_class_{i}.pkl')
        
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
        max_cluster = np.argmax(cluster_sizes)
        print('max_cluster:', max_cluster)

        probs = gmm.predict_proba(object_act)[:, max_cluster]
        selected_indices = np.argsort(probs)[::-1][:400]
        
        #print(selected_indices)
        result_indices = [index_dict[i][idx] for idx in selected_indices]

        results[i] = result_indices

    return results


def save_filtered_images(base_dataset, filtered_indices):
    grid_cols = len(filtered_indices[0])//10
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
    np.save(f'{DATASET_ROOT}quickdraw_npy/filtered_dataset_1.npy', filtered_dataset)

base_dataset = np.load(f'{DATASET_ROOT}quickdraw_npy/full_numpy_bitmap_all_objs.npy')
print(base_dataset.shape)
folder_name = "filtered_quickdraw"
checkpoint_folder_path = '/home/bpw10/mlrdev/MLR-2.0_clean2/checkpoints/VSSReady' #f'checkpoints/{folder_name}/'
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d=1, draw=True)
vae.eval()

filtered_indices = filter_quickdraw(vae, base_dataset, n_clusters=60)
save_filtered_images(base_dataset, filtered_indices)

