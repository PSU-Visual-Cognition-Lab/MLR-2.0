# prerequisites
import torch
from MLR_src.mVAE import train
import torch.optim as optim

def train_mVAE(dataloaders, vae, epoch_count, checkpoint_folder, use_wandb, start_epoch = 1, dimensions = []):
    if use_wandb is True:
        import wandb
        from MLR_src.wandb_setup import initialize_wandb, log_system_metrics
        initialize_wandb('2d-retina-train', {'version':'MLR_2.0_2D_RETINA_STN'})

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    seen_labels = {}
    components = 3*['shape'] + 3*['color']+ 3*['cropped'] + 3*['skip_cropped'] + ['retinal'] +  3*['object'] + 3*['cropped_object'] + ['retinal_object']

    for epoch in range(start_epoch, epoch_count):
        loss_lst, seen_labels = train(vae, optimizer, epoch, dataloaders, True, seen_labels, components, 600, checkpoint_folder)

        if use_wandb is True:
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
            'labels': seen_labels,
            'dimensions': dimensions
                    }
        if epoch % 4 == 0:
            torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
    
    if use_wandb is True:
        wandb.finish()