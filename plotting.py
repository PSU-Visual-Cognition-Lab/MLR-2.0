from simulation_src.plots_novel import cd_jiang_olson_chun_sim, cd_r_acc_vs_setsize, fig_loc_compare, load_checkpoint
import torch
import os

folder_name = 'test'

checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions
d = 1
vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', d)
device = torch.device(f'cuda:{d}')
torch.cuda.set_device(d)
print('checkpoint loaded')

run_name = 'test'
simulation_folder_path = f'simulations/{run_name}/'
if not os.path.exists('simulations/'):
    os.mkdir('simulations/')
    
if not os.path.exists(simulation_folder_path):
    os.mkdir(simulation_folder_path)

x = fig_loc_compare(vae, simulation_folder_path)