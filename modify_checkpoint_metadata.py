import torch
import argparse


parser = argparse.ArgumentParser(description="modifying checkpoint metadata")
parser.add_argument("--folder", type=str, default='test', help="Where to store checkpoints in checkpoints/")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--z_shape", type=int, default=8, help="Shape latent dimension")
parser.add_argument("--z_color", type=int, default=8, help="Color latent dimension")
parser.add_argument("--z_object", type=int, default=8, help="Object latent dimension")
parser.add_argument("--cuda_device", type=int, default=1, help="Which cuda device to use")
args = parser.parse_args()

d = args.cuda_device
filepath = f'checkpoints/{args.folder}/{args.checkpoint_name}'
    
if torch.cuda.is_available():
    device = torch.device(f'cuda:{d}')
    torch.cuda.set_device(d)
else:
    device = 'cpu'

torch_version = torch.__version__
if torch_version == '2.4.0':
    checkpoint = torch.load(filepath, device, weights_only = True)
else:
    checkpoint = torch.load(filepath, device)

old_dimensions = checkpoint['dimensions']
print(f'old dimensions: {old_dimensions}')
dimensions = [old_dimensions[0], old_dimensions[1], old_dimensions[2], args.z_shape, args.z_color, args.z_object]
print(f'new dimensions: {dimensions}')
checkpoint =  {
            'state_dict': checkpoint['state_dict'],
            'labels': checkpoint['labels'],
            'dimensions': dimensions,
            'training_components': checkpoint['training_components']
                    }

torch.save(checkpoint, f'checkpoints/{args.folder}/{args.checkpoint_name}')