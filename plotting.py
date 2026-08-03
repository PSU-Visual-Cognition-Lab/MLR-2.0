from simulation_src import figure_panels
from MLR_src.mVAE import load_checkpoint
from MLR_src.label_network import load_checkpoint_labels, s_classes, c_classes
import torch
import os
import matplotlib.pyplot as plt
import argparse
from joblib import load
import seaborn as sns

parser = argparse.ArgumentParser(description="Simulations using MLR-2.0")
parser.add_argument("--folder", type=str, default='test', help="where to find the vae checkpoint/")
parser.add_argument("--run_name", type=str, default='test', help="where to store simulation outputs/")
parser.add_argument("--panels", nargs='+', type=str, default=['all'], help="which figure panels to generate")
args = parser.parse_args()

# example terminal command given a checkpoint named "square_train_1" and a desired output folder "test_all":
# python plotting.py --folder square_train_1 --run_name test_all

folder_name = args.folder
run_name = args.run_name
panels = args.panels

checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions
d = 2
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d, True)
vae.eval()

vae_shape_labels = load_checkpoint_labels(f'{checkpoint_folder_path}/label_network_checkpoint.pth', "shape", d)
vae_object_labels = load_checkpoint_labels(f'{checkpoint_folder_path}/label_network_checkpoint.pth', "object", d)
vae_color_labels = load_checkpoint_labels(f'{checkpoint_folder_path}/label_network_checkpoint.pth', "color", d)

mnist_clf_shapeS = load(f'{checkpoint_folder_path}/mss.joblib')
emnist_clf_shapeS = load(f'{checkpoint_folder_path}/ess.joblib')
clf_objectS = load(f'{checkpoint_folder_path}/ooo.joblib')
clf_color = load(f'{checkpoint_folder_path}/ecc.joblib')
device = torch.device(f'cuda:{d}')
torch.cuda.set_device(d)
vae_color_labels.to(device)
vae_object_labels.to(device)
vae_shape_labels.to(device)
print('checkpoint loaded')

# set seaborn styles
sns.set_theme(context="paper", style="white")

simulation_folder_path = f'simulations/{run_name}/'
if not os.path.exists('simulations/'):
    os.mkdir('simulations/')
    
if not os.path.exists(simulation_folder_path):
    os.mkdir(simulation_folder_path)

panel_dict = {"synthesis": lambda: figure_panels.synthesis(vae, vae_shape_labels, s_classes, clf_objectS, simulation_folder_path),
              "individuated": lambda: figure_panels.individuated(vae, simulation_folder_path),
              "interference": lambda: figure_panels.interference(vae, simulation_folder_path),
              "basic": lambda: figure_panels.basic(vae, simulation_folder_path),
              "generative": lambda: figure_panels.generative(vae, vae_shape_labels, s_classes, vae_color_labels, c_classes, simulation_folder_path),
              "addressability": lambda: figure_panels.addressability(vae, clf_color, simulation_folder_path),
              "novel": lambda: figure_panels.novel(vae, simulation_folder_path),
              "compositional": lambda: figure_panels.compositional(vae, simulation_folder_path),
              "flexibility": lambda: figure_panels.flexibility(vae, simulation_folder_path),
              "modality": lambda: figure_panels.modality(vae, vae_shape_labels, s_classes, simulation_folder_path),
              "scene": lambda: figure_panels.scene(vae, vae_object_labels, vae_color_labels, clf_objectS, clf_color, simulation_folder_path),
              "poster": lambda: figure_panels.poster(vae, simulation_folder_path)
              }

if 'all' in panels:
    print('generating all figure panels')
    for name, panel_func in panel_dict.items():
        print(f'generating panel: {name}')
        panel_func()

elif panels:
    for panel in panels:
        if panel in panel_dict:
            print(f'generating panel: {panel}')
            panel_dict[panel]()
        else:
            print(f'panel {panel} not recognized')

'''figure_panels.synthesis(vae, vae_shape_labels, s_classes, clf_objectS, simulation_folder_path)
figure_panels.individuated(vae, simulation_folder_path)  #generating and retrieving specific examples of objects
figure_panels.interference(vae, simulation_folder_path)
figure_panels.basic(vae, simulation_folder_path)
figure_panels.generative(vae, vae_shape_labels, s_classes, vae_color_labels, c_classes, simulation_folder_path)
figure_panels.addressability(vae, clf_color, simulation_folder_path)
figure_panels.novel(vae, simulation_folder_path)
figure_panels.compositional(vae, simulation_folder_path)
figure_panels.flexibility(vae, simulation_folder_path)
figure_panels.poster(vae, simulation_folder_path)'''

#synthesis(vae, vae_shape_labels, s_classes, clf_objectS, simulation_folder_path)
#poster(vae, simulation_folder_path, False)
#individuated(vae, simulation_folder_path)
#interference(vae, simulation_folder_path)
#novel(vae, simulation_folder_path)
#addressability(vae, clf_color, simulation_folder_path)
#flexibility(vae, simulation_folder_path)
'''generative(vae, vae_shape_labels, s_classes, vae_color_labels, c_classes, simulation_folder_path)

synthesis(vae, vae_shape_labels, s_classes, clf_objectS, simulation_folder_path)

'''
#individuated(vae, simulation_folder_path)  #generating and retrieving specific examples of objects

#interference(vae, simulation_folder_path)
#basic(vae, simulation_folder_path)
#generative(vae, vae_shape_labels, s_classes, vae_color_labels, c_classes, simulation_folder_path)
#addressability(vae, clf_color, simulation_folder_path)

#novel(vae, simulation_folder_path)

#compositional(vae, simulation_folder_path)
#flexibility(vae, simulation_folder_path)

#novel(vae, simulation_folder_path)
