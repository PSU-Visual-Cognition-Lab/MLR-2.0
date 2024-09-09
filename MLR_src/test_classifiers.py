from classifiers import classifier_shape_test
from joblib import dump, load
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder
from mVAE import load_checkpoint

v=''
clf_ss = load(f'classifier_output{v}/ss.joblib')
clf_sc = load(f'classifier_output{v}/sc.joblib')
load_checkpoint(f'output_emnist_recurr{v}/checkpoint_300.pth') # MLR2.0 trained on emnist letters, digits, and fashion mnist

bs = 1001
test_bs = 1000
emnist_dataset, emnist_skip, emnist_test_dataset = dataset_builder('emnist', bs, None, False, None, False, True)
mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', bs, None, False, None, False, True)
test_loader = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_test_dataset, mnist_test_dataset]), batch_size=test_bs, shuffle=True,  drop_last= True)

pred_ss, pred_sc, SSreport, SCreport = classifier_shape_test('cropped', clf_ss, clf_sc, test_loader, 1)
print('accuracy:')
print('SS:',SSreport)
print('SC:',SCreport)