# prerequisites
import torch
import wandb
from MLR_src.mVAE import train
import torch.optim as optim
from MLR_src.wandb_setup import initialize_wandb, log_system_metrics

def train_mVAE(dataloaders, vae, epoch_count, checkpoint_folder):
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    initialize_wandb('2d-retina-train', {'version':'MLR_2.0_2D_RETINA'})

    seen_labels = {}
    components = ['color'] + ['shape'] + ['cropped'] * 2 + ['skip_cropped'] + ['location'] * 2

    for epoch in range(218, epoch_count):
        if epoch >= 200:
            components = ['location', 'retinal', 'retinal']
        elif epoch >= 500:
            components = ['color'] + ['shape'] + ['cropped'] + ['skip_cropped'] + ['location']
        elif epoch >= 560:
            components = ['retinal', 'retinal']

        loss_lst, seen_labels = train(vae, optimizer, epoch, dataloaders, True, seen_labels, components)

        wandb.log({
        'epoch': epoch,
        'retinal/training_loss': loss_lst[0],
        'retinal/test_loss': loss_lst[1],
        'cropped/training_loss': loss_lst[2],
        'cropped/test_loss': loss_lst[3]
        })

        torch.cuda.empty_cache()
        vae.eval()
        checkpoint =  {
            'state_dict': vae.state_dict(),
            #'labels': seen_labels
                    }
        torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
    
    wandb.finish()
