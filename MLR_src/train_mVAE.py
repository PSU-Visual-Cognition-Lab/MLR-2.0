# prerequisites
from xml.parsers.expat import model

import torch
from MLR_src.mVAE import train
import torch.optim as optim
from itertools import cycle
from training_constants import training_components

def get_trainable_components(model):
    return [
        name for name, module in model.named_children()
        if any(p.requires_grad for p in module.parameters())]

def train_mVAE(dataloaders, components, vae, epoch_count, checkpoint_folder, use_wandb, start_epoch = 1, dimensions = [], ep_size = 600, train_retinal_end = 0.1):
    if use_wandb is True:
        import wandb
        from MLR_src.wandb_setup import initialize_wandb, log_system_metrics
        initialize_wandb('final-training', {'version':'MLR_2.0_2D_RETINA_STN'}, checkpoint_folder)

    # seperate the learning rate for spatial transformer/ rest of network
    optimizer = optim.Adam(vae.parameters(), lr=0.00005, weight_decay=0.01)
    seen_labels = {}
    freeze_components = []

    # this logic trains the object level vae first, then freezes it to train the retina
    #components_no_skip = [s for s in components if "skip" not in s  ]
    components_no_ret = [s for s in components if "retina" not in s ]
    components_ret = [s for s in components if "retina" in s ]

    for epoch in range(start_epoch, epoch_count):
        #components_list = components_ret
        if epoch > train_retinal_end * epoch_count: # 
            components_list = components_no_ret
            freeze_components = ['localization', 'regressor']

        else:
            components_list = components_ret
            freeze_components = []
            #freeze_components = [x for x in get_trainable_components(vae) if x not in ['localization', 'regressor']]

        if epoch > 5 and epoch_count >= 1000:
            save_imgs = epoch % 50 == 0
        else:
            save_imgs = True
        
        loss_dicts, seen_labels = train(vae, optimizer, epoch, dataloaders, use_wandb, seen_labels, components_list, ep_size, freeze_components, checkpoint_folder, save_imgs)

        if use_wandb is True:   #this connects with weights and biases.. a website that tracks loss data over time.
            wandb_log = {'epoch': epoch}

            for phase, losses in loss_dicts.items():
                for name, value in losses.items():
                    wandb_log[f"{name}/{phase}_loss"] = value

            wandb.log(wandb_log)
            log_system_metrics()

        torch.cuda.empty_cache()
        
        vae.eval()
        checkpoint =  {
            'state_dict': vae.state_dict(),
            'labels': seen_labels,
            'dimensions': dimensions,
            'training_components': training_components
                    }

        '''if epoch == int((train_retinal_start * epoch_count) + 5):
            torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint2.pth')
        elif epoch == int((train_retinal_start * epoch_count) + 10):
            torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint2.pth')'''
        if epoch % 4 == 0:
            torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
    
    if use_wandb is True:
        wandb.finish()