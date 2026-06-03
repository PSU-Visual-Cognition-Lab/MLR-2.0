# current dataset builder

# prerequisites
import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms as torch_transforms
from torch.utils import data #.data import #DataLoader, Subset, Dataset
import random
import math
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION, ImageDraw

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

def _load_memmap(path: str) -> np.memmap:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[1] == 784:
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (28, 28):
        return arr
    raise ValueError(f"Unexpected shape in {path}: {arr.shape}")

def _ensure_28x28_uint8(a: np.ndarray) -> np.ndarray:
    return a.reshape(28, 28) if a.ndim == 1 else a

class RandomRotate90:
    """
    Rotate a PIL image or torch Tensor by k * 90°   with k ∈ {0,1,2,3}.
    Works before or after ToTensor().
    """
    def __init__(self, p=1.0):
        self.p = p                                    # 1.0 = always, <1 makes it optional

    def __call__(self, img):
        if torch.rand(1).item() > self.p:             # optional skip
            return img
        k = torch.randint(0, 4, ()).item()            # 0,1,2,3
        return torch_transforms.functional.rotate(img, angle=90 * k)

class Colorize_specific:
    def __init__(self, col):
        self.col = col

    def __call__(self, img):
        img = img.convert("L")
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
            scale_val = (random.random()*2) + 3
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
            return torch_transforms.ToTensor()(new_img), torch_transforms.ToTensor()(img), position  #retinal, crop, position

class ToTensor:
    def __init__(self):
        self.x = None
    def __call__(self, img):
        return torch_transforms.ToTensor()(img)

def generate_square_crop_image(image_size=(28, 28)):
    """Generate a black image with a white square in the center"""
    square_size = random.randint(7, 10)
    
    # Create a black background as a numpy array
    image_array = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    
    # Calculate center position for the square
    x = int((image_size[0] - square_size) // 2) + random.randint(-3,3)
    y = int((image_size[1] - square_size) // 2) + random.randint(-3,3)
    
    # Draw the white square (255 for white)
    image_array[y:y+square_size, x:x+square_size] = 255
    
    # Convert to PIL Image
    image = Image.fromarray(image_array, mode='L')
    
    return image

def generate_offset_line_crop_image(image_size=(28, 28)):
    """Generate a black image with a white line segment at random angle intervals of 18 degrees"""

    image_array = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    
    # Line parameters
    line_length = 20
    line_width = 2  # Changed from 3 to 2
    
    # Calculate center of image
    center_x = image_size[0] // 2
    center_y = image_size[1] // 2
    
    # Generate random angle from 10 possible angles (0, 18, 36, ..., 162 degrees)
    angle_step = 18
    angle_index = random.randint(0, 9)
    angle_degrees = angle_index * angle_step
    angle_radians = math.radians(angle_degrees)
    
    # Calculate start and end points of the line
    half_length = line_length // 2
    x1 = center_x - int(half_length * math.cos(angle_radians))
    y1 = center_y - int(half_length * math.sin(angle_radians))
    x2 = center_x + int(half_length * math.cos(angle_radians))
    y2 = center_y + int(half_length * math.sin(angle_radians))
    
    # For 2-pixel width, we'll use just two offset lines (one at -0.5 and one at +0.5)
    for offset in [-0.5, 0.5]:
        # Calculate perpendicular offset
        dx = int(offset * math.cos(angle_radians + math.pi/2))
        dy = int(offset * math.sin(angle_radians + math.pi/2))
        
        # Draw the offset line using Bresenham's algorithm
        x, y = x1 + dx, y1 + dy
        x2_offset, y2_offset = x2 + dx, y2 + dy
        
        dx = abs(x2_offset - x)
        dy = abs(y2_offset - y)
        steep = dy > dx
        
        if steep:
            x, y = y, x
            x2_offset, y2_offset = y2_offset, x2_offset
            dx, dy = dy, dx
            
        if x > x2_offset:
            x, x2_offset = x2_offset, x
            y, y2_offset = y2_offset, y
            
        gradient = dy/dx if dx != 0 else 1
        
        # Handle first endpoint
        xend = round(x)
        yend = y + gradient * (xend - x)
        xgap = 1 - ((x + 0.5) - int(x + 0.5))
        xpxl1 = xend
        ypxl1 = int(yend)
        
        if steep:
            if 0 <= ypxl1 < image_size[0] and 0 <= xpxl1 < image_size[1]:
                image_array[xpxl1, ypxl1] = 255
        else:
            if 0 <= xpxl1 < image_size[0] and 0 <= ypxl1 < image_size[1]:
                image_array[ypxl1, xpxl1] = 255
                
        intery = yend + gradient
        
        # Handle second endpoint
        xend = round(x2_offset)
        yend = y2_offset + gradient * (xend - x2_offset)
        xgap = (x2_offset + 0.5) - int(x2_offset + 0.5)
        xpxl2 = xend
        ypxl2 = int(yend)
        
        # Main line drawing loop
        for x in range(xpxl1 + 1, xpxl2):
            if steep:
                if 0 <= int(intery) < image_size[0] and 0 <= x < image_size[1]:
                    image_array[x, int(intery)] = 255
            else:
                if 0 <= x < image_size[0] and 0 <= int(intery) < image_size[1]:
                    image_array[int(intery), x] = 255
            intery += gradient
    
    # Convert to PIL Image
    image = Image.fromarray(image_array, mode='L')
    
    return image, angle_degrees

UPPERCASE_LABEL_INDICES = list(range(10,36))
LABEL_REMAP = {label: i for i, label in enumerate(UPPERCASE_LABEL_INDICES)}

class UppercaseEMNIST(torch.utils.data.Dataset):
    

    def __init__(self, root='./data/', train=True, transform=None):
        self.root = root
        self.train = train
        INDEX_CACHE_PATH = os.path.join(root, 'emnist_uppercase_indices.pt')

        self.base = datasets.EMNIST(
            root=self.root,
            split='balanced',
            train=self.train,
            download=False,
            transform=torch_transforms.Compose([lambda img: torch_transforms.functional.rotate(img, -90),
            lambda img: torch_transforms.functional.hflip(img)])
        )

        # Cache file includes 'train' in name to avoid collisions
        cache_file = INDEX_CACHE_PATH.replace('.pt', f'_train{int(train)}.pt')

        if os.path.exists(cache_file):
            self.indices = torch.load(cache_file)
        else:
            print('indexing EMNIST dataset...')
            self.indices = [
                i for i, (_, label) in enumerate(self.base)
                if label in LABEL_REMAP
            ]
            torch.save(self.indices, cache_file)
            print(f"Cached filtered indices to {cache_file}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        return img, LABEL_REMAP[label]

# the Dataset class defined below inherits the standard PyTorch data.Dataset class but modifies the __getitem__ method to apply
# complex transformations on request. the init of the Dataset class handles the necessary multi-transformation logic depending on
# the transforms input dict. all_possible_labels returns all valid feature combinations given the specified transformations.

class Dataset(data.Dataset):
    def __init__(self, dataset, transforms={}, train=True):
        # Use if the stimulus will put a stimulus into the retina
        if 'retina' in transforms: 
            self.retina = transforms['retina']

            if self.retina == True:

                if 'retina_size' in transforms:
                    self.retina_size = transforms['retina_size']

                else:
                    self.retina_size = 64

                if 'location_targets' in transforms:  #put some targets only on the left or right during training
                    print('inhere')
                    self.right_targets = transforms['location_targets']['right']
                    self.left_targets = transforms['location_targets']['left']

                else:
                    self.right_targets = []
                    self.left_targets = []
                
                if 'build_retina' in transforms:   #I'm not sure what this does, perhaps used when imagery creates a new form in the retina
                    self.build_ret = transforms['build_retina']
                else:
                    self.build_ret = True

            else:  #this will just be a cropped representation
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

            if self.scale == True:
                self.scale_dict = {}
                if 'scale_targets' in transforms:
                    for scale in transforms['scale_targets']:
                        for target in transforms['scale_targets'][scale]:
                            self.scale_dict[target] = scale

        else:
            self.scale = False
        
        #initialize rotation:
        if 'rotate' in transforms:
            self.rotate = transforms['rotate']
        
        else:
            self.rotate = False

        # initialize skip connection
        if 'skip' in transforms:
            self.skip = transforms['skip']

            if self.skip == True:
                self.colorize = True
                self.retina = False
        else:
            self.skip = False
        
        # initialize pair classes for quickdraw_pairs
        if 'pair_classes' in transforms:
            self.pair_classes = transforms['pair_classes']
        else:
            self.pair_classes = None
        
        if 'target_set' in transforms:
            self.target_set = transforms['target_set']
        else:
            self.target_set = None

        # initialize base dataset
        if type(dataset) == str:
            self.name = dataset
            self.train = train
            self.dataset = self._build_dataset(dataset, train)
            #self.data_source = self

        else:
            raise ValueError('invalid dataset input type')

        self.no_color_3dim = No_Color_3dim()
        self.totensor = ToTensor()
        self.target_dict = {'mnist':[0,9], 'emnist':[10,35], 'fashion_mnist':[36,45], 'cifar10':[0,9]} #[46,55]

    def _build_dataset(self, dataset, train=True):
        DATASET_ROOT = '/home/bwyble/data/'
        if dataset == 'mnist':
            base_dataset = datasets.MNIST(root=DATASET_ROOT, train=train, transform = None, download=True)

        elif dataset == 'emnist':
            split = 'letters' #by_class
            # raw emnist dataset is rotated and flipped by default, the applied transforms undo that
            '''DATA_PATH = './data'
            SAVE_PATH = os.path.join(DATA_PATH, 'emnist_uppercase.pt')
            if not os.path.exists(SAVE_PATH):
                print('Filtering EMNIST dataset')
                process_and_save_uppercase_emnist()'''
            
            base_dataset = UppercaseEMNIST(DATASET_ROOT)

            #base_dataset = datasets.EMNIST(root='./data', split=split, train=train, transform=torch_transforms.Compose([lambda img: torch_transforms.functional.rotate(img, -90),
            #lambda img: torch_transforms.functional.hflip(img)]), download=True)

        elif dataset == 'fashion_mnist':
            base_dataset = datasets.FashionMNIST(f'{DATASET_ROOT}fashionmnist_data/', train=train, transform = None, download=False)

        elif dataset == 'cifar10':
            base_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, download=False, transform=None)
        
        elif dataset == 'square':
            base_dataset = None
        
        elif dataset == 'line':
            base_dataset = None
        
        elif dataset == 'quickdraw':
            base_dataset = np.load(f'{DATASET_ROOT}quickdraw_npy/filtered_dataset_1.npy') #full_numpy_bitmap_all_objs
        
        elif dataset == 'quickdraw_pairs':
            base_dataset = {}
            for c in self.pair_classes:
                c = c[0]
                p = f"{DATASET_ROOT}quickdraw/{c}.npy"
                base_dataset[c] = _load_memmap(p)
            for c in self.pair_classes:
                c = c[1]
                p = f"{DATASET_ROOT}quickdraw/{c}.npy"
                base_dataset[c] = _load_memmap(p)

        elif os.path.exists(dataset):
            base_dataset = Image.open(rf'{dataset}')

        else:
            raise ValueError(f'{dataset} is not a valid base dataset')

        return base_dataset

    def _rng(self, index: int):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        s = (123 + 7919 * wid + 97 * index) & 0xFFFFFFFF
        return np.random.default_rng(s)

    def __len__(self):
        if self.dataset is None:
            return 10000

        elif type(self.dataset) != Image.Image:
            return len(self.dataset)

        else:
            return 10000

    def __getitem__(self, index):
        if self.name == 'square':
            image = generate_square_crop_image()
            target = -1
        
        elif self.name == 'line':
            image, target = generate_offset_line_crop_image()
        
        elif self.name == 'quickdraw':
            image = Image.fromarray(self.dataset[index, :-1].reshape(28, 28))  # image
            target = int(self.dataset[index, -1])  # label
        
        elif self.name == 'quickdraw_pairs':
            class_pair = random.choice(self.pair_classes)
            pair_idx = index % len(self.pair_classes)
            l_name, r_name = self.pair_classes[pair_idx]
            la, ra = self.dataset[l_name], self.dataset[r_name]

            rng = self._rng(index)
            li = rng.integers(0, la.shape[0])
            ri = rng.integers(0, ra.shape[0])

            l = _ensure_28x28_uint8(la[li])
            r = _ensure_28x28_uint8(ra[ri])

            l = torch.from_numpy(l).unsqueeze(0)  # (1,28,28) uint8
            r = torch.from_numpy(r).unsqueeze(0)  # (1,28,28) uint8
            image = torch.cat([l, r], dim=2) # (1,56,28) uint8
            image_np = image.squeeze(0).numpy()   # (56,28)
            image = Image.fromarray(image_np, mode='L')
            target = pair_idx  # label

        elif type(self.dataset) != Image.Image:
            image, target = self.dataset[index]
            if self.name == 'emnist' and self.train == True:
                #image = torch_transforms.ToPILImage()(image)
                pass
            else:
                target += self.target_dict[self.name][0]
            
        else:
            image = self.dataset
            target = 1
        
        if self.target_set is not None:
            if target not in self.target_set:
                new_idx = random.randint(2,len(self))-2
                return self.__getitem__(new_idx)

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
        elif self.name == 'cifar10':
            col = -1
        else:
            col = -1
            transform_list += [self.no_color_3dim]

        # skip connection dataset
        if self.skip == True:
            transform_list += [torch_transforms.RandomRotation(90), torch_transforms.RandomCrop(size=28, padding= 8)]
        
        if self.name == 'cifar10':
            resize = torch_transforms.Resize((28, 28))
            image =  resize(image)# resize 32,32 -> 28,28
        
        if self.rotate == True:
            transform_list += [RandomRotate90()]

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
        loader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, sampler=data.RandomSampler(self), drop_last=True)
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