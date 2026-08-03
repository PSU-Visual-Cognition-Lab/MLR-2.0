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
from skimage import color as skcolor
import warnings
from torch.utils.data import Subset
import joblib
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

L_MIN = 50.0   # darkest pixel: this L* value
L_MAX = 70.0   # brightest pixel: this L* value
AB_VARIATION_SCALE = 50.0

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

def _rgb_to_lab(rgb_0_1):
    arr = np.array(rgb_0_1, dtype=np.float64).reshape(1, 1, 3)
    return skcolor.rgb2lab(arr).reshape(3)

def _max_chroma_scale(a, b, l_min=L_MIN):
    """
    Binary-search for the largest scale s in [0,1] such that the Lab colour
    (l_min, a*s, b*s) converts to a non-negative sRGB triplet.
    The worst case is always the darkest pixel (l_min), so we only need to
    check that one L* value.
    """
    if a == 0 and b == 0:
        return 1.0
    lo, hi = 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):          # 20 iterations → precision < 1e-6
            mid = (lo + hi) / 2.0
            lab = np.array([[[l_min, a * mid, b * mid]]])
            rgb = skcolor.lab2rgb(lab)
            if rgb.min() >= 0.0:
                lo = mid
            else:
                hi = mid
    return lo

colorvals_lab = []   # (L*, a*, b*, chroma_scale)
 
for rgb in colorvals:
    L, a, b = _rgb_to_lab(rgb)
    scale = _max_chroma_scale(a, b)
    colorvals_lab.append((L, a, b, scale))

class Colorize_specific:
    """
    Drop-in replacement for Colorize_specific that works in CIELAB.
 
    Parameters
    ----------
    col : int
        Index into colornames / colorvals_lab (same index as before).
    l_min, l_max : float
        Luminance range [0, 100] to which the grayscale is linearly mapped.
    ab_variation_scale : float
        Scales the ±colorrange random variation into a*/b* units.
    """
 
    def __init__(
        self,
        col: int,
        l_min: float = L_MIN,
        l_max: float = L_MAX,
        ab_variation_scale: float = AB_VARIATION_SCALE,
    ):
        self.col = col
        self.l_min = l_min
        self.l_max = l_max
        self.ab_variation_scale = ab_variation_scale
 
    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. convert to 8-bit grayscale
        gray = np.array(img.convert("L"), dtype=np.float64)   # [0, 255]
        H, W = gray.shape
 
        # 2. normalise grayscale → L* in [l_min, l_max]
        g_norm = gray / 255.0
        L = self.l_min + g_norm * (self.l_max - self.l_min)
 
        # 3. sample a*/b* near the base colour centroid
        _, base_a, base_b, chroma_scale = colorvals_lab[self.col]
 
        # apply gamut scale first, then add variation within the scaled space
        a_center = base_a * chroma_scale
        b_center = base_b * chroma_scale
        var = colorrange * self.ab_variation_scale
        a_val = a_center + np.random.uniform(-var, var)
        b_val = b_center + np.random.uniform(-var, var)
 
        # 4. build full Lab image  (H × W × 3)
        lab_img = np.stack(
            [L, np.full((H, W), a_val), np.full((H, W), b_val)],
            axis=-1,
        )
 
        # 5. convert Lab → sRGB; clip residual float errors only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_float = skcolor.lab2rgb(lab_img).clip(0.0, 1.0)
 
        rgb_uint8 = (rgb_float * 255).astype(np.uint8)
        bg_mask = gray < 10          # tweak threshold as needed
        rgb_uint8[bg_mask] = 0
        return Image.fromarray(rgb_uint8, "RGB")

class Colorize_specific_RGB:
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

class Translate_old:
    # TODO replace loc with (x, y) vector
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

def axis_padding(sign, size, max_width, min_width=28):
    if sign not in (-1, 1):
        raise ValueError(f"quadrant must be a tuple of (-1|1, -1|1), got {sign}")

    half = max_width // 2

    if size > max_width:
        raise ValueError(
            f"object size {size} exceeds retina width {max_width}; "
            f"this should have been caught by _validate_scale."
        )

    if sign == -1:
        if size <= half:
            low, high = 0, half - size
        else:
            low, high = 0, 0
    else:
        if size <= half:
            low, high = half, max_width - size
        else:
            low = high = max_width - size

    return int(random.uniform(low, high))

class Translate:
    def __init__(self, scale_value, padding, max_width, min_width=28, build_ret=True):
        self.max_width = max_width
        self.min_width = min_width
        self.max_scale = max_width // 2
        self.pos = torch.zeros(2, max_width)
        self.padding_left = padding[0]
        self.padding_bottom = padding[1]
        self.scale_value = scale_value
        self.build_ret = build_ret

    def _validate_scale(self, img):
        """
        Ensures self.scale_value applied to img's native size fits within
        the retina. If not, warns and returns the largest valid scale for
        this specific image's native size instead.
        """
        width = img.size[0] * self.scale_value
        height = img.size[1] * self.scale_value

        if width <= self.max_width and height <= self.max_width:
            return self.scale_value

        max_valid_scale = self.max_width / max(img.size[0], img.size[1])
        warnings.warn(
            f"scale_value={self.scale_value} on image size {img.size} would produce "
            f"({width:.1f}, {height:.1f}), exceeding retina max_width={self.max_width}. "
            f"Clamping to max valid scale={max_valid_scale:.4f} for this image."
        )
        return max_valid_scale

    def __call__(self, img):
        scale_value = self._validate_scale(img)

        width = int(img.size[0] * scale_value)
        height = int(img.size[1] * scale_value)
        img = torch_transforms.Resize((height, width))(img)

        padding_left = self.padding_left #self._axis_padding(self.quadrant[0], img.size[0])
        padding_right = self.max_width - img.size[0] - padding_left

        padding_bottom = self.padding_bottom# self._axis_padding(self.quadrant[1], img.size[1])
        padding_top = self.max_width - img.size[1] - padding_bottom

        assert 0 <= padding_left <= self.max_width - img.size[0]
        assert 0 <= padding_bottom <= self.max_width - img.size[1]

        pos = self.pos.clone()
        pos[0][padding_left] = 1
        pos[1][padding_bottom] = 1

        if self.build_ret is False:
            return 0, pos

        padding = (padding_left, padding_top, padding_right, padding_bottom)
        return ImageOps.expand(img, padding), pos

class PadAndPosition:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        new_img, position = self.transform(img)
        return torch_transforms.ToTensor()(new_img), torch_transforms.ToTensor()(img), position #retinal, crop, position

class ToTensor:
    def __init__(self):
        self.x = None
    def __call__(self, img):
        return torch_transforms.ToTensor()(img)

def generate_square_crop_image(image_size=(28, 28)):
    """Generate a black image with a white square in the center"""
    square_size = 16
    
    # Create a black background as a numpy array
    image_array = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    
    # Calculate center position for the square
    x = int((image_size[0] - square_size) // 2)
    y = int((image_size[1] - square_size) // 2)
    
    # Draw the white square (255 for white)
    image_array[y:y+square_size, x:x+square_size] = 255
    
    # Convert to PIL Image
    image = Image.fromarray(image_array, mode='L')
    
    return image

def generate_noise_patch_image(image_size=(28, 28), density=0.7):
    """Generate a black image with a random noise patch in a random location"""
    
    image_array = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    
    patch_h = np.random.randint(16, image_size[0])
    patch_w = np.random.randint(16, image_size[1])
    
    y = np.random.randint(0, image_size[0] - patch_h + 1)
    x = np.random.randint(0, image_size[1] - patch_w + 1)
    
    noise_patch = np.random.randint(0, 256, size=(patch_h, patch_w), dtype=np.uint8)
    if density < 1.0:
        density_mask = np.random.random((patch_h, patch_w)) < density
        noise_patch = noise_patch * density_mask
    image_array[y:y+patch_h, x:x+patch_w] = noise_patch
    
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

def load_filtered_dataset(root, train=True, tag='mnist'):
    base = datasets.MNIST(root=root, train=train, transform=None, download=True)
    filtered_indices = joblib.load(f'{root}{tag}_filtered_indices.pkl')
    
    # Flatten dict of {class_id: [indices]} into a single index list
    all_indices = [idx for indices in filtered_indices.values() for idx in indices]
    
    return Subset(base, all_indices)

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

    @classmethod
    def load_filtered(cls, root, train=True, tag='emnist'):
        # Instantiate normally to get the full uppercase subset
        instance = cls(root=root, train=train)

        # Load the GMM-filtered base indices {class_id: [base_indices]}
        filtered = joblib.load(os.path.join(root, f'{tag}_filtered_indices.pkl'))

        # Flatten to a set for O(1) lookup
        filtered_base_indices = set(
            idx for indices in filtered.values() for idx in indices
        )

        # self.indices maps UppercaseEMNIST positions -> base indices
        # Keep only positions whose base index passed the GMM filter
        instance.indices = [
            base_idx for base_idx in instance.indices
            if base_idx in filtered_base_indices
        ]

        return instance

# the Dataset class defined below inherits the standard PyTorch data.Dataset class but modifies the __getitem__ method to apply
# complex transformations on request. the init of the Dataset class handles the necessary multi-transformation logic depending on
# the transforms input dict. all_possible_labels returns all valid feature combinations given the specified transformations.
TARGETS = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, # mnist
                        'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P': 25, 'Q':26, 'R':27, 'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35, # emnist
                        't-shirt':36, 'trouser':37, 'pullover':38, 'dress':39, 'coat':40, 'sandal':41, 'shirt':42, 'sneaker':43, 'bag':44, 'ankle boot':45, #fashion mnist
                        'airplane':46, 'automobile':47, 'bird':48, 'cat':49, 'deer':50, 'dog':51, 'frog':52, 'horse':53, 'ship':54, 'truck':55, # cifar10
                        'square':56}

def label_to_string(label: int) -> str:
    inverted_labels = {v: k for k, v in TARGETS.items()}
    return inverted_labels.get(label, "Unknown")


class Dataset(data.Dataset):
    def __init__(self, dataset, transforms={}, train=True):
        # Use if the stimulus will put a stimulus into the retina
        if 'retina' in transforms: 
            self.retina = transforms['retina']
            self.scale_range = {} # init

            if self.retina == True:

                if 'retina_size' in transforms:
                    self.retina_size = transforms['retina_size']

                else:
                    self.retina_size = 64

                if 'location_targets' in transforms:
                    #print('inhere')
                    # transforms['location_targets'] := {(-1,-1): [0,1,2], (1,1): [3,4,5]...}
                    if 'right' in transforms['location_targets'] or 'left' in transforms['location_targets']:
                        ValueError('right and left location targets have been deprecated, use one of the following quadrants: (1,1), (-1,1), (1,-1), (-1,-1) to specify the quadrant in which the target should be placed')                       

                    self.target_quadrants = {}
                    for quadrant in transforms['location_targets']:
                        if quadrant not in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            raise ValueError(f'quadrant {quadrant} is not valid, must be one of: (-1,-1), (-1,1), (1,-1), (1,1)')

                        for target in transforms['location_targets'][quadrant]:
                            self.target_quadrants[target] = quadrant

                else:
                    self.target_quadrants = {}
                
                if 'build_retina' in transforms:   #I'm not sure what this does, perhaps used when imagery creates a new form in the retina
                    self.build_ret = transforms['build_retina']
                else:
                    self.build_ret = True

            else:  #this will just be a cropped representation
                self.retina_size = None
                self.target_quadrants = {}

        else:
            self.retina = False
            self.retina_size = None
            self.target_quadrants = {}

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

                if 'scale_range' in transforms:
                    self.scale_range = transforms['scale_range']

        else:
            self.scale_range = {}
        
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
        
        self.target_dict = {'mnist':[0,9], 'emnist':[10,35], 'fashion_mnist':[36,45], 'cifar10':[46,55], 'square':[56, 56]} #[46,55]

    def _build_dataset(self, dataset, train=True):
        DATASET_ROOT = '/home/bwyble/data/'
        if dataset == 'mnist':
            base_dataset = load_filtered_dataset(DATASET_ROOT, train=train, tag='mnist')

        elif dataset == 'emnist':
            split = 'letters' #by_class
            # raw emnist dataset is rotated and flipped by default, the applied transforms undo that
            '''DATA_PATH = './data'
            SAVE_PATH = os.path.join(DATA_PATH, 'emnist_uppercase.pt')
            if not os.path.exists(SAVE_PATH):
                print('Filtering EMNIST dataset')
                process_and_save_uppercase_emnist()'''
            
            base_dataset = UppercaseEMNIST.load_filtered(DATASET_ROOT, train=train, tag='emnist')

            #base_dataset = datasets.EMNIST(root='./data', split=split, train=train, transform=torch_transforms.Compose([lambda img: torch_transforms.functional.rotate(img, -90),
            #lambda img: torch_transforms.functional.hflip(img)]), download=True)

        elif dataset == 'fashion_mnist':
            base_dataset = datasets.FashionMNIST(f'{DATASET_ROOT}fashionmnist_data/', train=train, transform = None, download=False)

        elif dataset == 'cifar10':
            base_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, download=False, transform=None)
        
        elif dataset == 'square':
            base_dataset = None

        elif dataset == 'noise_mask':
            base_dataset = None
        
        elif dataset == 'line':
            base_dataset = None
        
        elif dataset == 'quickdraw':
            base_dataset = np.load(f'{DATASET_ROOT}quickdraw_npy/filtered_dataset_1.npy')

        elif dataset == 'quickdraw_full': # unfiltered quickdraw dataset
            base_dataset = np.load(f'{DATASET_ROOT}quickdraw_npy/full_numpy_bitmap_all_objs.npy')
        
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
            target = 56

        elif self.name == 'noise_mask':
            image = generate_noise_patch_image()
            target = 56

        elif self.name == 'line':
            image, target = generate_offset_line_crop_image()
        
        elif self.name in ['quickdraw', 'quickdraw_full']:
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
            target += self.target_dict[self.name][0]
            
        else:
            image = self.dataset
            target = 0
        
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
                    if len(self.scale_range) != 2:
                        scale = 1
                    else:
                        bins = np.linspace(self.scale_range[0], self.scale_range[1], 20)
                        scale = random.choice(bins)
            else:
                scale = 1

            if target in self.target_quadrants:
                translation_quadrant = self.target_quadrants[target]
            else:
                #print('TARGET MISS', target, self.target_quadrants)
                translation_quadrant = random.choice([(-1, -1), (1, 1), (-1, 1), (1, -1)])
            if image.size != (28,28):
                print('size miss')
            width = int(image.size[0] * scale)
            height = int(image.size[1] * scale)
            left_padding = axis_padding(translation_quadrant[0], width, self.retina_size)
            bottom_padding = axis_padding(translation_quadrant[1], height, self.retina_size)
            top_padding = self.retina_size - height - bottom_padding
            translation = (left_padding, bottom_padding)
            translation_label = (left_padding + width / 2, top_padding + height / 2)
            translate = PadAndPosition(Translate(scale, translation, self.retina_size, self.build_ret))
            transform_list += [translate]
        else:
            scale = 1
            translation_label = (-1, -1) # no translation
            transform_list += [self.totensor]

        # labels: target, color, x_translation, y_translation, scale
        out_label = (target, col, translation_label[0], translation_label[1], scale)
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
                scale = 1
                translation = (9, 9) # no translation

            # labels
            target = [col, translation, scale]
            target_dict[i] = target

        return target_dict