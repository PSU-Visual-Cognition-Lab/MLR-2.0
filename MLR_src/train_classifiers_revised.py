# prerequisites

import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import utils
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt
from joblib import dump
import joblib
'''
dataset options: 
emnist-map
mnist-map
emnist-skip
quickdraw
mnist-skip
'''

def train_classifiers(dataloaders, vae, checkpoint_folder):

    '''    print('training emnist data on color map using color labels')   #this should be high
    clf_ecc = classifier_train(vae, 'color', dataloaders['emnist-map'], 'color')
    dump(clf_ecc, f'checkpoints/{checkpoint_folder}/ecc.joblib')
    pred_ecc, ecc_report = classifier_test(vae, 'color', clf_ecc, dataloaders['emnist-map'],'emnist','color', 1)
    
    print('training emnist data on shape map using shape labels')   #this should be high
    clf_ess = classifier_train(vae, 'shape', dataloaders['emnist-map'], 'shape')
    dump(clf_ess, f'checkpoints/{checkpoint_folder}/ess.joblib')
    pred_ess, ess_report = classifier_test(vae, 'shape', clf_ess, dataloaders['emnist-map'],'emnist','shape', 1)

    print('training mnist data on shape map using shape labels')   #this should be high
    clf_mss = classifier_train(vae, 'shape', dataloaders['mnist-map'], 'shape')
    dump(clf_mss, f'checkpoints/{checkpoint_folder}/mss.joblib')
    pred_mss, mss_report = classifier_test(vae, 'shape', clf_mss, dataloaders['mnist-map'],'mnist','shape', 1)
    '''
    print('training object data on OBJECT map using object labels')  #this should be high if object latent is working
    clf_ooo = classifier_train(vae, 'object', dataloaders['quickdraw-map'],'object')
    dump(clf_ooo, f'checkpoints/{checkpoint_folder}/ooo.joblib')
    pred_ooo, ooo_report = classifier_test(vae, 'object', clf_ooo, dataloaders['quickdraw-map'],'quickdraw','object', 1)

    print('training object data on color map using color labels')   #this should be high
    clf_occ = classifier_train(vae, 'color', dataloaders['quickdraw-map'], 'color')
    dump(clf_occ, f'checkpoints/{checkpoint_folder}/occ.joblib')
    pred_occ, occ_report = classifier_test(vae, 'color', clf_occ, dataloaders['quickdraw-map'],'quickdraw','color', 1)

    print('training object data on color map using object labels')   #this should be low but above chance
    clf_oco = classifier_train(vae, 'color', dataloaders['quickdraw-map'],'object')
    dump(clf_oco, f'checkpoints/{checkpoint_folder}/oco.joblib')
    pred_oco, oco_report = classifier_test(vae, 'color', clf_oco, dataloaders['quickdraw-map'],'quickdraw','object' ,1)

    # ==================== DIAGNOSTIC CROSS-DECODING TESTS ====================
    # These tests check whether object identity leaks into the shape latent
    # and whether letter identity leaks into the object latent.
    # If the latents are properly separated, both should be near chance.

    print('\n===== CROSS-DECODING DIAGNOSTICS =====')

    print('DIAGNOSTIC: object data on SHAPE map using object labels')  # should be LOW if routing is fixed
    clf_oso = classifier_train(vae, 'shape', dataloaders['quickdraw-map'], 'object')
    dump(clf_oso, f'checkpoints/{checkpoint_folder}/oso.joblib')
    pred_oso, oso_report = classifier_test(vae, 'shape', clf_oso, dataloaders['quickdraw-map'], 'quickdraw', 'object', 1)

    print('DIAGNOSTIC: emnist data on OBJECT map using shape labels')  # should be LOW if routing is fixed
    clf_eos = classifier_train(vae, 'object', dataloaders['emnist-map'], 'shape')
    dump(clf_eos, f'checkpoints/{checkpoint_folder}/eos.joblib')
    pred_eos, eos_report = classifier_test(vae, 'object', clf_eos, dataloaders['emnist-map'], 'emnist', 'shape', 1)

    print('\n===== SUMMARY =====')
    print(f'Letters from SHAPE map  (should be HIGH): {ess_report:.4f}')
    print(f'Object from OBJECT map  (should be HIGH): {ooo_report:.4f}')
    print(f'Object from SHAPE map   (should be LOW):  {oso_report:.4f}')
    print(f'Letters from OBJECT map (should be LOW):  {eos_report:.4f}')
    print('=====================================')



def classifier_train(vae, whichcomponent, train_dataset, whichlabel):
    #trains an svm using a given dataset, from a given latent space (whichcomponent) and datalabel

    vae.eval()
    clf = svm.SVC(C=10, gamma='scale', kernel='rbf', probability=True)

    labelindex = 0  #shape/object by default
    if whichlabel == 'color':
        labelindex = 1

    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        data = data[1]
        train_labels = labels[labelindex].clone()
        utils.save_image(data[0:10], 'train_sample.png')
        data = data.to(device)
        activations = vae.activations(data, False, None)
        z = activations[whichcomponent].to(device)  # now uses 'object' key directly when whichcomponent=='object'
        clf.fit(z.cpu().numpy(), train_labels.cpu())
    return clf

def classifier_test(vae, whichcomponent, clf, test_dataset, dataname, whichlabel, verbose=0):
    vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    vae.eval()
    device = next(vae.parameters()).device

    labelindex = 0  #shape/object by default
    if whichlabel == 'color':
        labelindex = 1

    with torch.no_grad():
        data, labels = next(iter(test_dataset))
        data = data[1]
        test_labels = labels[labelindex].clone()
        data = data.cuda()
        activations = vae.activations(data, False, None)
        z = activations[whichcomponent].to(device)  # now uses 'object' key directly when whichcomponent=='object'
        pred = clf.predict(z.cpu())
        test_labels = test_labels.cpu().numpy()

        report = accuracy_score(pred, test_labels)
     
        if verbose == 1:
            cm = confusion_matrix(test_labels, pred)

            print(f'test accuracy: {whichlabel} classification from the {whichcomponent} map using dataset {dataname}')
            print(confusion_matrix(test_labels, pred))
            print(classification_report(test_labels, pred))
            print('accuracy:')
            print(report)
            

    return pred, report
