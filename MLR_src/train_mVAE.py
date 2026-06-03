# prerequisites
import torch
from MLR_src.mVAE import train
import torch.optim as optim
from itertools import cycle
from training_constants import training_components

def train_mVAE(dataloaders, components, vae, epoch_count, checkpoint_folder, use_wandb, start_epoch = 1, dimensions = []):
    if use_wandb is True:
        import wandb
        from MLR_src.wandb_setup import initialize_wandb, log_system_metrics
        initialize_wandb('final-training', {'version':'MLR_2.0_2D_RETINA_STN'}, checkpoint_folder)

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    seen_labels = {}
        
    #components_no_skip = [s for s in components if "skip" not in s  ]
    components_no_ret = [s for s in components if "retina" not in s ]
    for epoch in range(start_epoch, epoch_count):
        if epoch > 7:
            components_list = components_no_ret
        else:
            components_list = components

        loss_dicts, seen_labels = train(vae, optimizer, epoch, dataloaders, True, seen_labels, components_list, 600, checkpoint_folder)

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
        if epoch % 4 == 0:
            torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
    
    if use_wandb is True:
        wandb.finish()