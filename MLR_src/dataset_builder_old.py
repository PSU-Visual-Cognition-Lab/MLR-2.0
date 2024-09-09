# prerequisites
import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms as torch_transforms
from torch.utils import data #.data import #DataLoader, Subset, Dataset
from random import randint
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION

colornames = ["red", "green", "blue", "purple", "yellow", "cyan", "orange", "brown", "pink", "white"]
colorrange = .08
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [1-colorrange,1-colorrange,1-colorrange]
]

class Colorize_specific:
    def __init__(self, col):   
        self.col = col

    def __call__(self, img):
        # col: an int index for which base color is being used
        rgb = colorvals[self.col]  # grab the rgb for this base color

        r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
        g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
        b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
        np_img = np_img.astype(np.uint8)
        img = Image.fromarray(np_img, 'RGB')

        return img

class No_Color_3dim:
    def __init__(self):
        self.x = None
        
    def __call__(self, img):
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        np_img = np_img.astype(np.uint8)
        img = Image.fromarray(np_img, 'RGB')
        return img

class Translate:
    def __init__(self, loc, max_width):
        self.max_width = max_width
        self.pos = torch.zeros((100))
        self. loc = loc
    def __call__(self, img):
        if self.loc == 1:
            padding_left = randint(0, (self.max_width // 2) - img.size[0])
            padding_right = self.max_width - img.size[0] - padding_left
        elif self.loc == 2:
            padding_left = randint(self.max_width // 2, self.max_width - img.size[0])
            padding_right = self.max_width - img.size[0] - padding_left   

        padding = (padding_left, 0, padding_right, 0)
        pos = self.pos.clone()
        pos[padding_left] = 1
        return ImageOps.expand(img, padding), pos

class PadAndPosition:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img):
        new_img, position = self.transform(img)
        return torch_transforms.ToTensor()(new_img), torch_transforms.ToTensor()(img), position

class ToTensor:
    def __init__(self):
        self.x = None
    def __call__(self, img):
        return torch_transforms.ToTensor()(img)
        
class Dataset(data.Dataset):
    def __init__(self, dataset, transforms={}, train=True):
        # initialize base dataset
        if type(dataset) == str:
            self.name = dataset
            self.train = train
            self.dataset = self._build_dataset(dataset, train)

        else:
            raise ValueError('invalid dataset input type')
        
        # initialize retina
        if 'retina' in transforms:
            self.retina = transforms['retina']

            if self.retina == True:

                if 'retina_size' in transforms:
                    self.retina_size = transforms['retina_size']
                
                else:
                    self.retina_size = 100
                
                if 'location_targets' in transforms:
                    self.right_targets = transforms['location_targets']['right']
                    self.left_targets = transforms['location_targets']['left']
                
                else:
                    self.right_targets = []
                    self.left_targets = []
       
            else:
                self.retina_size = None
                self.right_targets = []
                self.left_targets = []
        
        else:
            self.retina = False
            self.retina_size = None
            self.right_targets = []
            self.left_targets = []

        # initialize colors
        if 'colorize' in transforms:
            self.colorize = transforms['colorize']
        
        else:
            self.colorize = False
        
        if 'color_targets' in transforms:
            self.color_dict = None
            colors = {}
            for color in transforms['color_targets']:
                for target in transforms['color_targets'][color]:
                    colors[target] = color

            self.color_dict = colors
        
        else:
            self.color_dict = {}
        
        # initialize skip connection
        if 'skip' in transforms:
            self.skip = transforms['skip']

            if self.skip == True:
                self.colorize = True
                self.retina = False
        else:
            self.skip = False
        
        self.no_color_3dim = No_Color_3dim()
        self.totensor = ToTensor()
        self.target_dict = {'mnist':[0,9], 'emnist':[10,35], 'fashion_mnist':[36,45], 'cifar10':[46,55]}        

        if dataset == 'emnist':
            self.lowercase = list(range(0,10)) + list(range(36,63))
            if os.path.exists('uppercase_ind_train.pt'):
                if self.train == True:
                    self.indices = torch.load('uppercase_ind_train.pt')
                else:
                    self.indices = torch.load('uppercase_ind_test.pt') 
            else:
                print('indexing emnist dataset:')
                self.indicies, self.indices = self._filter_indices()
                print('indexing complete')
    
    def _filter_indices(self):
        base_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, transform=torch_transforms.Compose([lambda img: torch_transforms.functional.rotate(img, -90),
            lambda img: torch_transforms.functional.hflip(img)]), download=True)
        indices_test = []
        count = {target: 0 for target in list(range(10,36))}
        print('starting indices collection')
        for i in range(len(base_dataset)):
            img, target = base_dataset[i]
            if target not in self.lowercase and count[target] <= 6000:
                indices_test += [i]
                count[target] += 1
        print(count)
        #torch.save(indices_train, 'uppercase_ind_train.pt')
        torch.save(indices_test, 'uppercase_ind_test.pt')
        print('saved indices')
        indices_train = torch.load('uppercase_ind_train.pt')
        return indices_train, indices_test

    def _build_dataset(self, dataset, train=True):
        if dataset == 'mnist':
            base_dataset = datasets.MNIST(root='./mnist_data/', train=train, transform = None, download=True)

        elif dataset == 'emnist':
            split = 'byclass'
            # raw emnist dataset is rotated and flipped by default, the applied transforms undo that
            base_dataset = datasets.EMNIST(root='./data', split=split, train=train, transform=torch_transforms.Compose([lambda img: torch_transforms.functional.rotate(img, -90),
            lambda img: torch_transforms.functional.hflip(img)]), download=True)

        elif dataset == 'fashion_mnist':
            base_dataset = datasets.FashionMNIST('./fashionmnist_data/', train=train, transform = None, download=True)

        elif dataset== 'cifar10':
            base_dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=None)

        elif os.path.exists(dataset):
            pass

        else:
            raise ValueError(f'{dataset} is not a valid base dataset')

        return base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        if self.name == 'emnist' and self.train == True:
            image, target = self.dataset[self.indices[randint(0,len(self.indices)-1)]]
        else:
            target += self.target_dict[self.name][0]
        col = None
        transform_list = []
        # append transforms according to transform attributes
        # color
        if self.colorize == True:
            if target in self.color_dict:
                col = self.color_dict[target]
                transform_list += [Colorize_specific(col)]
            else:
                col = randint(0,9) # any
                transform_list += [Colorize_specific(col)]
        else:
            col = -1
            transform_list += [self.no_color_3dim]

        # skip connection dataset
        if self.skip == True:
            transform_list += [torch_transforms.RandomRotation(90), torch_transforms.RandomCrop(size=28, padding= 8)]

        # retina
        if self.retina == True:
            if target in self.left_targets:
                translation = 1 # left  
            elif target in self.right_targets:
                translation = 2 # right
            else:
                translation = randint(1,2) #any
            translate = PadAndPosition(Translate(translation, self.retina_size))
            transform_list += [translate]
        else:
            translation = -1
            transform_list += [self.totensor]

        # labels
        out_label = (target, col, translation)
        transform = torch_transforms.Compose(transform_list)
        return transform(image), out_label

    def get_loader(self, batch_size):
        loader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=True,  drop_last=True)
        return loader
    
    def all_possible_labels(self):
        # return a list of all possible labels generated by this dataset in order: (shape identity, color, retina location)
        dataset = self.name
        start = self.target_dict[dataset][0]
        end = self.target_dict[dataset][1] + 1
        target_dict = {}

        for i in range(start,end):
            if self.colorize == True:
                if i in self.color_dict:
                    col = [self.color_dict[i]]
                else:
                    col = [0,9]
            else:
                col = [-1]

            # retina
            if self.retina == True:
                if i in self.left_targets:
                    translation = [1]
                elif i in self.right_targets:
                    translation = [2]
                else:
                    translation = [1,2]
            else:
                translation = [-1]

            # labels
            target = [col, translation]
            target_dict[i] = target
        
        return target_dict