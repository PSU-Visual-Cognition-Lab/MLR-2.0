from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms, utils

training_related_pairs = [('airplane', 'bird'), ('tree', 'axe')]
testing_related_pairs = [('airplane', 'bird')]
testing_unrelated_pairs = [('airplane', 'axe')]

quickdraw_target_set = [0,2,8,10,11]
location_targets = {(-1,-1):[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], (1,1):[0,1,2,3,4,5,6,7,8,9]}
#  'location_targets': location_targets
scale_min = 0.75
scale_max = 2

# dataset names must be in format <dataset name>-<component type>, unless there is only one component trained by that dataset
training_datasets = {'emnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'scale_range':[scale_min, scale_max]},
                     'emnist-color_bg_map': {'retina':True, 'colorize':True, 'colorize_background':True, 'rotate':False, 'scale':True, 'scale_range':[scale_min, scale_max]},
                     'emnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True, 'scale_range':[scale_min, scale_max]},
                     'mnist-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'scale_range':[scale_min, scale_max]},
                     'mnist-skip': {'retina':False, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True, 'scale_range':[scale_min, scale_max]},
                     'quickdraw-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'target_set': quickdraw_target_set, 'scale_range':[scale_min, scale_max]},
                     'quickdraw-color_bg_map': {'retina':True, 'colorize':True, 'colorize_background':True, 'rotate':False, 'scale':True, 'target_set': quickdraw_target_set, 'scale_range':[scale_min, scale_max]},
                     'quickdraw-skip': {'retina':True, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True, 'scale_range':[scale_min, scale_max]},
                     'quickdraw_full-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'target_set': quickdraw_target_set, 'scale_range':[scale_min, scale_max]},
                     'quickdraw_full-skip': {'retina':True, 'colorize':True, 'rotate':True, 'scale':True, 'skip': True, 'scale_range':[scale_min, scale_max]},
                     'cifar10': {'retina':True, 'rotate':False, 'scale':True},
                     'square-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'scale_range':[scale_min, scale_max]},
                     'noise_mask-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'scale_range':[scale_min, scale_max]},
                     'line': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'fashion_mnist': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True},
                     'quickdraw_pairs-map': {'retina':True, 'colorize':True, 'rotate':False, 'scale':True, 'class_pairs': training_related_pairs}}

training_components = {'shape': [['emnist-map', 'mnist-map', 'emnist-color_bg_map'], 2], # shape map, weighted 2 times in training etc #'emnist-map', 'emnist-map', 'mnist-map', 'square-map'
                       'color': [['emnist-map', 'mnist-map', 'quickdraw-map', 'emnist-color_bg_map'], 3], # color map 
                       'object': [['quickdraw-map', 'quickdraw-color_bg_map'], 1], # map for quickdraw
                       'cropped': [['emnist-map', 'mnist-map', 'emnist-color_bg_map'], 2], # shape and color recon
                       'cropped_object': [['quickdraw-map', 'quickdraw-color_bg_map'], 1], # object and color recon
                       'skip_cropped': [['emnist-skip', 'mnist-skip', 'quickdraw-skip'], 1], # mnist/emnist skip connection
                       'retinal': [['emnist-map', 'mnist-map', 'square-map', 'noise_mask-map', 'emnist-color_bg_map'], 1], #   retinal, scale, location
                       'retinal_object': [['quickdraw_full-map','square-map', 'noise_mask-map', 'quickdraw-color_bg_map'], 1]} #   retinal, scale, location, object

def text_to_tensor(text,height,width):
    img = Image.new('RGB', (width, height), (255, 255, 255))
    ImageDraw.Draw(img).text((10, 10), text, fill=(0, 0, 0))
    return transforms.ToTensor()(img)