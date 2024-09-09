# prerequisites
import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms as torch_transforms
from torch.utils import data #.data import #DataLoader, Subset, Dataset
import random
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
    def __init__(self, scale, loc, max_width, min_width = 28, build_ret = True):
        self.max_width = max_width
        self.min_width = min_width
        self.max_scale = max_width//2
        self.pos = torch.zeros(2, max_width).cuda()
        self.loc = loc
        self.scale = scale
        self.build_ret = build_ret

    def __call__(self, img):
        if self.scale == 0:
            scale_val = (random.random()*4)
            scale_dist = torch.zeros(10)
            scale_dist[int(scale_val)] = 1
            width = int(self.min_width + (self.max_width - self.min_width) * (scale_val / 10))
            height = int(self.min_width + (self.max_width - self.min_width) * (scale_val/ 10))
            resize = torch_transforms.Resize((width, height))
            img = resize(img)

        elif self.scale == 1:
            scale_val = (random.random()*4) +4
            scale_dist = torch.zeros(10)
            scale_dist[int(scale_val)] = 1
            width = int(self.min_width + (self.max_width - self.min_width) * (scale_val / 10))
            height = int(self.min_width + (self.max_width - self.min_width) * (scale_val/ 10))
            resize = torch_transforms.Resize((width, height))
            img = resize(img)

        else:
            scale_dist = None

        if self.loc == 1:
            padding_left = int(random.uniform(0, (self.max_width // 2)-(img.size[0]//2))) #include center overlap region +
            padding_right = self.max_width - img.size[0] - padding_left
            padding_bottom = random.randint(0, self.max_width - img.size[0])
            padding_top = self.max_width - img.size[0] - padding_bottom

        elif self.loc == 2:
            if img.size[0] >= self.max_width//2:
              x = img.size[0]//2
            else:
              x = 0
            padding_left = int(random.uniform((self.max_width // 2)-x, self.max_width - img.size[0])) #include center overlap region
            padding_right = self.max_width - img.size[0] - padding_left
            padding_bottom = random.randint(0, self.max_width - img.size[0])
            padding_top = self.max_width - img.size[0] - padding_bottom
        
        pos = self.pos.clone()
        pos[0][padding_left] = 1
        pos[1][padding_bottom] = 1
        
        if self.build_ret is False:
            return 0, pos, scale_dist
        
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        #print(padding_left,padding_bottom)
        return ImageOps.expand(img, padding), pos, scale_dist

class PadAndPosition:
    def __init__(self, transform):
        self.transform = transform
        self.scale = transform.scale
    def __call__(self, img):
        new_img, position, scale_dist = self.transform(img)
        if self.scale != -1:
            return torch_transforms.ToTensor()(new_img), torch_transforms.ToTensor()(img), position, scale_dist #retinal, crop, position, scale
        else:
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
                    self.retina_size = 64

                if 'location_targets' in transforms:
                    self.right_targets = transforms['location_targets']['right']
                    self.left_targets = transforms['location_targets']['left']

                else:
                    self.right_targets = []
                    self.left_targets = []
                
                if 'build_retina' in transforms:
                    self.build_ret = transforms['build_retina']
                else:
                    self.build_ret = True

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
            self.color_dict = {}

            if self.colorize == True and 'color_targets' in transforms:
                self.color_dict = {}
                colors = {}
                for color in transforms['color_targets']:
                    for target in transforms['color_targets'][color]:
                        colors[target] = color

                self.color_dict = colors

        else:
            self.colorize = False
            self.color_dict = {}

        # initialize scaling
        if 'scale' in transforms:
            self.scale = transforms['scale']

            if self.scale == True and 'scale_targets' in transforms:
                self.scale_dict = {}
                for scale in transforms['scale_targets']:
                    for target in transforms['scale_targets'][scale]:
                        self.scale_dict[target] = scale

        else:
            self.scale = False

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
            base_dataset = Image.open(rf'{dataset}')

        else:
            raise ValueError(f'{dataset} is not a valid base dataset')

        return base_dataset

    def __len__(self):
        if type(self.dataset) != Image.Image:
            return len(self.dataset)
        else:
            return 1

    def __getitem__(self, index):
        if type(self.dataset) != Image.Image:
            image, target = self.dataset[index]
            if self.name == 'emnist' and self.train == True:
                image, target = self.dataset[self.indices[random.randint(0,len(self.indices)-1)]]
            else:
                target += self.target_dict[self.name][0]
        else:
            image = self.dataset
            target = 1
        col = None
        transform_list = []
        # append transforms according to transform attributes
        # color
        if self.colorize == True:
            if target in self.color_dict:
                col = self.color_dict[target]
                transform_list += [Colorize_specific(col)]
            else:
                col = random.randint(0,9) # any
                transform_list += [Colorize_specific(col)]
        else:
            col = -1
            transform_list += [self.no_color_3dim]

        # skip connection dataset
        if self.skip == True:
            transform_list += [torch_transforms.RandomRotation(90), torch_transforms.RandomCrop(size=28, padding= 8)]

        # retina
        if self.retina == True:
            if self.scale == True:
                if target in self.scale_dict:
                    scale = self.scale_dict[target]
                else:
                    scale = random.randint(0,1)
            else:
                scale = -1

            if target in self.left_targets:
                translation = 1 # left
            elif target in self.right_targets:
                translation = 2 # right
            else:
                translation = random.randint(1,2) #any

            translate = PadAndPosition(Translate(scale, translation, self.retina_size, self.build_ret))
            transform_list += [translate]
        else:
            scale = -1
            translation = -1
            transform_list += [self.totensor]

        # labels
        out_label = (target, col, translation, scale)
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