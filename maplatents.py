# Visualize shape and object latent spaces using t-SNE
# Usage: python maplatents.py --folder test --components shape color object

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
import os
import sys

from sklearn.manifold import TSNE

from MLR_src.mVAE import load_checkpoint, load_dimensions, VAE_CNN
from MLR_src.dataset_builder import Dataset
from training_constants import training_components, training_datasets
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default='test', help="Where to find checkpoints in checkpoints/")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--components", nargs='+', type=str, default=['shape', 'color','object'], help="Which latent spaces to visualize")
parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to collect")
parser.add_argument("--cuda_device", type=int, default=0)
parser.add_argument("--use_mu", action='store_true', default=True, help="Use mu (True) or sampled z (False)")
args = parser.parse_args()

checkpoint_path = f'checkpoints/{args.folder}/{args.checkpoint_name}'
output_dir = f'latent_visualizations/{args.folder}/'

# setup
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.cuda_device}')
    torch.cuda.set_device(args.cuda_device)
else:
    device = 'cpu'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load model
vae = load_checkpoint(checkpoint_path, args.cuda_device, draw=True)
vae.eval()

# build dataloaders for each component
bs = 100
dataloaders = {}
for component in args.components:
    if component not in training_components:
        print(f"Skipping unknown component: {component}")
        continue
    print(f"Using component: {component}")

    for dataset_name in training_components[component][0]:
        if dataset_name not in dataloaders:
            base_name = dataset_name.split('-')[0]
            transforms = training_datasets[dataset_name]
            loader = cycle(Dataset(base_name, transforms).get_loader(bs))
            dataloaders[dataset_name] = iter(loader)

# collect latent activations and images
def collect_latents(vae: VAE_CNN, dataloaders, component, n_samples, use_mu=True):
    """Collect latent vectors, labels, and cropped images for a given component"""
    dataset_names = training_components[component][0]
    # use all datasets per component
    dataloaders = cycle([dataloaders[dataset_name] for dataset_name in dataset_names])
    
    all_latents = []
    all_shape_labels = []
    all_color_labels = []
    all_images = []
    collected = 0

    while collected < n_samples:
        dataloader = next(dataloaders)
        data, labels = next(dataloader)
        if type(data) == list:
            cropped_image = data[1].to(device)
            retinal_image = data[0].to(device)
        else:
            image = data.to(device)

        # store raw image tensors for centroid visualization
        all_images.append(cropped_image.cpu())

        with torch.no_grad():
            cropped_activations = vae.activations(cropped_image, False)
            cropped_shape_latent = cropped_activations['shape']
            cropped_color_latent = cropped_activations['color']
            cropped_object_latent = cropped_activations['object']

            retinal_activations = vae.activations(retinal_image, True)
            retinal_shape_latent = retinal_activations['shape']
            retinal_color_latent = retinal_activations['color']
            retinal_object_latent = retinal_activations['object']

        if component == 'object':
            z = retinal_object_latent # torch.cat([cropped_object_latent, retinal_object_latent], 0) 
        elif component == 'shape':
            z = cropped_shape_latent#torch.cat([cropped_shape_latent, retinal_shape_latent], 0)
        elif component == 'color':
            z = cropped_color_latent#torch.cat([cropped_color_latent, retinal_color_latent], 0)
        else:
            print(f"Unknown component: {component}")
            return None, None, None, None
        
        all_latents.append(z.cpu().numpy())
        all_shape_labels.append(labels[0].numpy())
        all_color_labels.append(labels[1].numpy())
        collected += len(cropped_image)

    latents = np.concatenate(all_latents, axis=0)[:n_samples]
    shape_labels = np.concatenate(all_shape_labels, axis=0)[:n_samples]
    color_labels = np.concatenate(all_color_labels, axis=0)[:n_samples]
    images = torch.cat(all_images, dim=0)[:n_samples]

    return latents, shape_labels, color_labels, images


# label name maps for readability
emnist_label_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                       10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
                       20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
                       30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 56: 'square'}
color_label_names = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow',
                     5: 'cyan', 6: 'orange', 7: 'brown', 8: 'pink', 9: 'white'}


def plot_embedding_with_centroids(embedding, labels, images, title, save_path,
                                   label_names=None, img_zoom=0.5):
    """Plot t-SNE with scatter points and sample images at each class centroid"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', n_labels) if n_labels <= 20 else plt.cm.get_cmap('nipy_spectral', n_labels)

    # draw scatter points
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names and label in label_names else str(label)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=5, alpha=0.3, label=name)

    # draw sample images at centroids
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_points = embedding[mask]

        # compute centroid in embedding space
        centroid = class_points.mean(axis=0)

        # find the sample closest to the centroid
        dists = np.linalg.norm(class_points - centroid, axis=1)
        closest_idx_in_class = np.argmin(dists)

        # map back to global index
        global_indices = np.where(mask)[0]
        closest_idx = global_indices[closest_idx_in_class]

        # get the image and convert to numpy HWC for display
        img_tensor = images[closest_idx]  # (3, 28, 28)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # place image at centroid
        imagebox = OffsetImage(img_np, zoom=img_zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (centroid[0], centroid[1]),
                            frameon=True,
                            bboxprops=dict(edgecolor=cmap(i), linewidth=2,
                                           facecolor='white', alpha=0.85),
                            pad=0.2)
        ax.add_artist(ab)

    ax.set_title(title, fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# main loop
for component in args.components:
    if component not in training_components:
        continue

    print(f"\nCollecting {args.n_samples} samples for {component} latent space...")
    latents, shape_labels, color_labels, images = collect_latents(
        vae, dataloaders, component, args.n_samples, args.use_mu)

    if latents is None:
        continue

    print(f"  Latent shape: {latents.shape}")
    print(f"  Unique shape labels: {np.unique(shape_labels)}")
    print(f"  Unique color labels: {np.unique(color_labels)}")

    # determine label names
    if component == 'shape':
        label_names = emnist_label_names
    else:
        label_names = None

    # t-SNE
    print(f"  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_embedding = tsne.fit_transform(latents)

    plot_embedding_with_centroids(
        tsne_embedding, shape_labels, images,
        f't-SNE of {component} latent (colored by identity)',
        os.path.join(output_dir, f'tsne_{component}_by_identity.png'),
        label_names)

    plot_embedding_with_centroids(
        tsne_embedding, color_labels, images,
        f't-SNE of {component} latent (colored by color)',
        os.path.join(output_dir, f'tsne_{component}_by_color.png'),
        color_label_names)

print("\nDone! Check", output_dir)
