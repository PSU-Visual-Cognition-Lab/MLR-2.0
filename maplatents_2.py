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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from MLR_src.mVAE import load_checkpoint, load_dimensions
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
# Subcluster parameters
parser.add_argument("--dbscan_eps", type=float, default=3.0,
                    help="DBSCAN eps (neighborhood radius in t-SNE space). "
                         "Increase to merge nearby clusters, decrease to split them.")
parser.add_argument("--dbscan_min_samples", type=int, default=10,
                    help="DBSCAN min_samples (minimum points to form a core point). "
                         "Increase to require denser clusters.")
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
def collect_latents(vae, dataloaders, component, n_samples, use_mu=True):
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
            image = data[1].to(device)
        else:
            image = data.to(device)

        # store raw image tensors for centroid visualization
        all_images.append(image.cpu())

        with torch.no_grad():
            if component == 'object':
                mu_object, log_var_object = vae.encoder_object(image)
                mu = mu_object
                log_var = log_var_object
            elif component == 'shape':
                mu_shape, log_var_shape, mu_color, log_var_color, hskip = vae.encoder(image)
                mu = mu_shape
                log_var = log_var_shape
            elif component == 'color':
                mu_shape, log_var_shape, mu_color, log_var_color, hskip = vae.encoder(image)
                mu = mu_color
                log_var = log_var_color
            else:
                print(f"Unknown component: {component}")
                return None, None, None, None

            if use_mu:
                z = mu
            else:
                z = vae.sampling(mu, log_var)

        all_latents.append(z.cpu().numpy())
        all_shape_labels.append(labels[0].numpy())
        all_color_labels.append(labels[1].numpy())
        collected += len(image)

    latents = np.concatenate(all_latents, axis=0)[:n_samples]
    shape_labels = np.concatenate(all_shape_labels, axis=0)[:n_samples]
    color_labels = np.concatenate(all_color_labels, axis=0)[:n_samples]
    images = torch.cat(all_images, dim=0)[:n_samples]

    return latents, shape_labels, color_labels, images


# label name maps for readability
emnist_label_names = {i: chr(58 + i) for i in range(0, 26)}  # 10=A, 11=B, ... 35=Z
color_label_names = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow',
                     5: 'cyan', 6: 'orange', 7: 'brown', 8: 'pink', 9: 'white'}


def find_subclusters(points_2d, eps=3.0, min_samples=10):
    """
    Use DBSCAN to find subclusters within a set of 2D t-SNE points.

    Returns an array of subcluster labels (same length as points_2d).
    Label -1 means noise/outlier — these points are excluded from centroid display.

    Tuning tips:
      - eps: the neighborhood radius in t-SNE space. Your axes run ~[-80, 80],
        so eps=3.0 is ~2% of the range — a reasonable default. Raise it (e.g. 5–8)
        if too many small fragments appear; lower it (e.g. 1–2) if blobs that look
        separate are being merged.
      - min_samples: minimum points to form a dense core. Raise it to suppress
        tiny noisy clusters; lower it if real clusters are being dropped as noise.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    sub_labels = db.fit_predict(points_2d)
    return sub_labels


def get_closest_to_centroid(points, global_indices, images):
    """Return the image tensor of the sample closest to the mean of `points`."""
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    closest_local = np.argmin(dists)
    closest_global = global_indices[closest_local]
    img_tensor = images[closest_global]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    return centroid, img_np


def plot_embedding_with_centroids(embedding, labels, images, title, save_path,
                                   label_names=None, img_zoom=0.5,
                                   dbscan_eps=3.0, dbscan_min_samples=10):
    """
    Plot t-SNE with:
      - Scatter points coloured by class label
      - A LARGE image box at each class centroid (overall centroid marker)
      - A SMALL image box at each DBSCAN subcluster centroid within every class
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = (plt.cm.get_cmap('tab20', n_labels)
            if n_labels <= 20
            else plt.cm.get_cmap('nipy_spectral', n_labels))

    # ── 1. Draw scatter points ───────────────────────────────────────────────
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names and label in label_names else str(label)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=5, alpha=0.3, label=name)

    # ── 2. Per-class: overall centroid + per-subcluster centroids ────────────
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_points = embedding[mask]          # (N_class, 2)
        global_indices = np.where(mask)[0]
        color = cmap(i)

        # -- 2a. Overall class centroid (large box) ---------------------------
        class_centroid, class_img = get_closest_to_centroid(
            class_points, global_indices, images)

        imagebox_large = OffsetImage(class_img, zoom=img_zoom)
        imagebox_large.image.axes = ax
        ab_large = AnnotationBbox(
            imagebox_large, (class_centroid[0], class_centroid[1]),
            frameon=True,
            bboxprops=dict(edgecolor=color, linewidth=3,
                           facecolor='white', alpha=0.95),
            pad=0.3,
            zorder=4,          # draw on top of subcluster boxes
        )
        ax.add_artist(ab_large)

        # -- 2b. Subcluster centroids (small boxes) ---------------------------
        sub_labels = find_subclusters(class_points,
                                      eps=dbscan_eps,
                                      min_samples=dbscan_min_samples)
        unique_sub = np.unique(sub_labels)
        unique_sub = unique_sub[unique_sub != -1]   # drop noise

        n_sub = len(unique_sub)
        if n_sub > 1:   # only draw if more than one subcluster found
            for sub_id in unique_sub:
                sub_mask_local = sub_labels == sub_id
                sub_points = class_points[sub_mask_local]
                sub_global = global_indices[sub_mask_local]

                sub_centroid, sub_img = get_closest_to_centroid(
                    sub_points, sub_global, images)

                imagebox_small = OffsetImage(sub_img, zoom=img_zoom * 0.6)
                imagebox_small.image.axes = ax
                ab_small = AnnotationBbox(
                    imagebox_small, (sub_centroid[0], sub_centroid[1]),
                    frameon=True,
                    bboxprops=dict(edgecolor=color, linewidth=1.5,
                                   facecolor='white', alpha=0.75,
                                   linestyle='dashed'),   # dashed = subcluster
                    pad=0.15,
                    zorder=3,
                )
                ax.add_artist(ab_small)

    # ── 3. Legend and labels ─────────────────────────────────────────────────
    from matplotlib.patches import Patch
    style_legend = [
        Patch(facecolor='white', edgecolor='grey', linewidth=2,
              label='Class centroid (solid border)'),
        Patch(facecolor='white', edgecolor='grey', linewidth=1.5,
              linestyle='dashed', label='Subcluster centroid (dashed border)'),
    ]

    class_legend = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                              markerscale=3, fontsize=8, ncol=2, title='Class')
    ax.add_artist(class_legend)
    ax.legend(handles=style_legend, bbox_to_anchor=(1.01, 0.12),
              loc='lower left', fontsize=8, title='Marker guide')

    ax.set_title(title, fontsize=16)
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
        label_names,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples)

    plot_embedding_with_centroids(
        tsne_embedding, color_labels, images,
        f't-SNE of {component} latent (colored by color)',
        os.path.join(output_dir, f'tsne_{component}_by_color.png'),
        color_label_names,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples)

print("\nDone! Check", output_dir)