# prerequisites
from classifiers import classifier_shape_train, classifier_color_train, clf_sc, clf_ss, clf_cc, clf_cs, classifier_shape_test, classifier_color_test
from mVAE import load_checkpoint
from joblib import dump, load
from dataset_builder import Dataset
#from torch.utils.data import DataLoader, ConcatDataset
import torch
import os

v = '2'
folder_path = f'classifier_output{v}' # the output folder for the trained model versions

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

load_checkpoint(f'output_mnist_2drecurr{v}/checkpoint_1700.pth') # MLR2.0 trained on emnist letters, digits, and fashion mnist

bs = 20000
test_bs = 10000
# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
#emnist_transforms = {'retina':True, 'colorize':True}
mnist_transforms = {'retina':False, 'colorize':True, 'scale':False, 'build_retina':False}
mnist_test_transforms = {'retina':False, 'colorize':True, 'scale':False}

#emnist_dataset = Dataset('emnist', emnist_transforms)
mnist_dataset = Dataset('mnist', mnist_transforms)

#emnist_test_dataset = Dataset('emnist', emnist_transforms, train= False)
mnist_test_dataset = Dataset('mnist', mnist_test_transforms, train= False)

#concat datasets and init dataloaders
train_loader = mnist_dataset.get_loader(bs)
test_loader = mnist_test_dataset.get_loader(test_bs)

print('training shape classifiers')
classifier_shape_train('cropped', train_loader)
dump(clf_sc, f'{folder_path}/sc.joblib')
dump(clf_ss, f'{folder_path}/ss.joblib')

pred_ss, pred_sc, SSreport, SCreport = classifier_shape_test('cropped', clf_ss, clf_sc, test_loader)
print('accuracy:')
print('SS:',SSreport)
print('SC:',SCreport)

print('training color classifiers')
classifier_color_train('cropped', train_loader)
dump(clf_cc, f'{folder_path}/cc.joblib')
dump(clf_cs, f'{folder_path}/cs.joblib')

pred_cc, pred_cs, CCreport, CSreport = classifier_color_test('cropped', clf_cc, clf_cs, test_loader)
print('accuracy:')
print('CC:',CCreport)
print('CS:',CSreport)