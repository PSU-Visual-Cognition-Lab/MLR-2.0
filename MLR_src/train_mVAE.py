# prerequisites
import torch
from MLR_src.mVAE import train
import torch.optim as optim

def train_mVAE(dataloaders, vae, epoch_count, checkpoint_folder, load=False):
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    if load is True:
        loss_dict = torch.load(f'../checkpoints/{checkpoint_folder}/loss_data.pth')
    else:
        loss_dict = {'retinal_train':[], 'retinal_test':[], 'cropped_train':[], 'cropped_test':[]}
    seen_labels = {}
    for epoch in range(0, epoch_count):
        loss_lst, seen_labels = train(vae, optimizer, epoch, dataloaders, True, seen_labels)
        
        # save error quantities
        loss_dict['retinal_train'] += [loss_lst[0]]
        loss_dict['retinal_test'] += [loss_lst[1]]
        loss_dict['cropped_train'] += [loss_lst[2]]
        loss_dict['cropped_test'] += [loss_lst[3]]
        torch.save(loss_dict, f'../checkpoints/{checkpoint_folder}/loss_data.pth')

        torch.cuda.empty_cache()
        vae.eval()
        checkpoint =  {
            'state_dict': vae.state_dict(),
            'labels': seen_labels
                    }
        torch.save(checkpoint, f'../checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
