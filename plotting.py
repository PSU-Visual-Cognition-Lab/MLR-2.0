from simulation_src.figure_panels import individuated, interference, novel, addressability, generative, synthesis, compositional, flexibility
from MLR_src.mVAE import load_checkpoint
from MLR_src.label_network import load_checkpoint_labels, s_classes, c_classes
import torch
import os
import matplotlib.pyplot as plt
import argparse
from joblib import load
import seaborn as sns

parser = argparse.ArgumentParser(description="Simulations using MLR-2.0")
parser.add_argument("--c_folder", type=str, default='test', help="where to find the vae checkpoint/")
parser.add_argument("--run_name", type=str, default='test', help="where to store simulation outputs/")
args = parser.parse_args()

# example terminal command given a checkpoint named "square_train_1" and a desired output folder "test_all":
# python plotting.py --c_folder square_train_1 --run_name test_all

folder_name = args.c_folder
run_name = args.run_name

checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions
d = 1
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d, True)
vae.eval()

vae_shape_labels = load_checkpoint_labels(f'{checkpoint_folder_path}/label_network_checkpoint.pth', "shape", d)
vae_color_labels = load_checkpoint_labels(f'{checkpoint_folder_path}/label_network_checkpoint.pth', "color", d)

mnist_clf_shapeS = load(f'{checkpoint_folder_path}/mss.joblib')
emnist_clf_shapeS = load(f'{checkpoint_folder_path}/ess.joblib')
clf_objectS = load(f'{checkpoint_folder_path}/ooo.joblib')
clf_color = load(f'{checkpoint_folder_path}/ecc.joblib')
device = torch.device(f'cuda:{d}')
torch.cuda.set_device(d)
vae_color_labels.to(device)
vae_shape_labels.to(device)
print('checkpoint loaded')

# set seaborn styles
sns.set_theme(context="paper", style="white")

simulation_folder_path = f'simulations/{run_name}/'
if not os.path.exists('simulations/'):
    os.mkdir('simulations/')
    
if not os.path.exists(simulation_folder_path):
    os.mkdir(simulation_folder_path)

#individuated(vae, simulation_folder_path)
#interference(vae, simulation_folder_path)
#novel(vae, simulation_folder_path)
#addressability(vae, clf_color, simulation_folder_path)
#flexibility(vae, simulation_folder_path)
'''generative(vae, vae_shape_labels, s_classes, vae_color_labels, c_classes, simulation_folder_path)
synthesis(vae, vae_shape_labels, s_classes, clf_objectS, simulation_folder_path)
#compositional(vae, simulation_folder_path)
#novel(vae, simulation_folder_path)

interference(vae, simulation_folder_path)'''
