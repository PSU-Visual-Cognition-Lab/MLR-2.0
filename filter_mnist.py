import numpy as np
from MLR_src.mVAE import load_checkpoint
from sklearn.mixture import GaussianMixture
import torch
from torchvision import datasets, transforms as torch_transforms
from torchvision.utils import save_image
from collections import defaultdict
import joblib
import os
import math
from PIL import Image
import numpy as np

DATASET_ROOT = '/home/bwyble/data/'
CACHE_DIR = 'data/mnist_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_mnist_by_class(dataset):
    """Group torchvision MNIST/EMNIST dataset indices by class label."""
    index_dict = defaultdict(list)
    for i, (_, label) in enumerate(dataset):
        index_dict[int(label)].append(i)
    return index_dict

@torch.no_grad()
def filter_mnist(model, dataset, n_clusters=10, d=1, classes_to_keep=None, top_k_ratio=0.8):
    transform = torch_transforms.Compose([
        torch_transforms.Grayscale(num_output_channels=3),  # VAE expects 3-channel
        torch_transforms.ToTensor(),
    ])
    # Re-load with transform (or pass in pre-transformed dataset)
    index_dict = get_mnist_by_class(dataset)
    
    if classes_to_keep is None:
        classes_to_keep = list(index_dict.keys())

    results = {}

    for class_id in classes_to_keep:
        print(f'Processing class {class_id}')
        act_cache_path = f'{CACHE_DIR}/object_act_class_{class_id}.pkl'

        if not os.path.exists(act_cache_path):
            indices = index_dict[class_id]
            object_act = []

            # Process in chunks to manage memory
            chunk_size = 1000
            for chunk_start in range(0, len(indices), chunk_size):
                chunk_indices = indices[chunk_start:chunk_start + chunk_size]
                samples = torch.stack([
                    transform(dataset[i][0])  # dataset[i] = (PIL image, label)
                    for i in chunk_indices
                ]).to(d)  # [N, 3, 28, 28]

                activations = model.activations(samples)
                object_act.append(activations['object'].cpu())

            object_act = torch.cat(object_act, dim=0)
            joblib.dump(object_act, act_cache_path)
        else:
            print(f'  Loading cached activations for class {class_id}')
            object_act = joblib.load(act_cache_path)

        print(f'  object_act shape: {object_act.shape}')

        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(object_act)

        cluster_sizes = np.bincount(labels)
        max_cluster = np.argmax(cluster_sizes)
        print(f'  Dominant cluster: {max_cluster} (size {cluster_sizes[max_cluster]})')

        probs = gmm.predict_proba(object_act)[:, max_cluster]

        # Clamp to actual activation count, not index_dict count (may differ)
        n_available = len(object_act)
        top_k = min(math.ceil(n_available * top_k_ratio), n_available)
        selected_local = np.argsort(probs)[::-1][:top_k]

        # Also guard the index mapping
        global_indices = [index_dict[class_id][i] for i in selected_local if i < len(index_dict[class_id])]
        results[class_id] = global_indices

    return results


def save_filtered_indices(filtered_indices, tag='mnist'):
    """Save only the index list — no image data duplication."""
    out_path = f'{DATASET_ROOT}{tag}_filtered_indices.pkl'
    joblib.dump(filtered_indices, out_path)
    print(f'Saved filtered indices to {out_path}')
    for class_id, indices in filtered_indices.items():
        print(f'  Class {class_id}: {len(indices)} samples')


def save_preview_grids(dataset, filtered_indices, tag='mnist', grid_cols=20, preview_cap=500):
    for class_id, indices in filtered_indices.items():
        preview_indices = indices[:preview_cap]  # already sorted best-first
        images = [np.array(dataset[i][0]) for i in preview_indices]
        n = len(images)
        grid_rows = math.ceil(n / grid_cols)
        # MNIST is grayscale — stack to RGB for consistency
        grid = np.zeros((grid_rows * 28, grid_cols * 28, 3), dtype=np.uint8)
        for k, img in enumerate(images):
            row, col = divmod(k, grid_cols)
            rgb = np.dstack([img, img, img]) if img.ndim == 2 else img
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = rgb
        Image.fromarray(grid, 'RGB').save(f'filtered_images/{tag}/class_{class_id}_grid.png')
        print(f'  Saved preview grid for class {class_id}')


# --- Main ---
checkpoint_folder_path = 'checkpoints/mlr-2-fulltest2'
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d=1, draw=True)
vae.eval()

# Load MNIST — images stay as PIL, transform applied inside filter_mnist
#mnist = datasets.MNIST(root=DATASET_ROOT, train=True, download=True, transform=None)
emnist = datasets.EMNIST(
            root=DATASET_ROOT,
            split='balanced',
            train=True,
            download=False,
            transform=torch_transforms.Compose([lambda img: torch_transforms.functional.rotate(img, -90),
            lambda img: torch_transforms.functional.hflip(img)]))

filtered_indices = filter_mnist(
    vae, emnist,
    n_clusters=60,
    classes_to_keep=None,
    top_k_ratio=0.7,
)

save_filtered_indices(filtered_indices, tag='emnist')
save_preview_grids(emnist, filtered_indices, tag='emnist')