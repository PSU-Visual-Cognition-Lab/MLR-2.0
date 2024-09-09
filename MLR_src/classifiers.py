# prerequisites
import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import utils
from mVAE import vae
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt

# defining the classifiers
clf_ss = svm.SVC(C=10, gamma='scale', kernel='rbf', probability= True)  # define the classifier for shape
clf_sc = svm.SVC(C=10, gamma='scale', kernel='rbf')  # classify shape map against color labels
clf_cc = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for color
clf_cs = svm.SVC(C=10, gamma='scale', kernel='rbf')  # classify color map against shape labels

vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#training the shape map on shape labels and color labels
def classifier_shape_train(whichdecode_use, train_dataset):
    vae.eval()
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()
        print(train_shapelabels[0:10])
        utils.save_image(data[0:10],'train_sample.png')

        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,sc,j = vae(data, whichdecode_use)
        z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
        print('training shape bottleneck against color labels sc')
        clf_sc.fit(z_shape.cpu().numpy(), train_colorlabels)

        print('training shape bottleneck against shape labels ss')
        clf_ss.fit(z_shape.cpu().numpy(), train_shapelabels)

#testing the shape classifier (one image at a time)
def classifier_shape_test(whichdecode_use, clf_ss, clf_sc, test_dataset, confusion_mat=0):
    vae.eval()
    with torch.no_grad():
        data, labels = next(iter(test_dataset))
        test_shapelabels=labels[0].clone()
        test_colorlabels=labels[1].clone()
        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,sc,j = vae(data, whichdecode_use)
        z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
        pred_ss = clf_ss.predict(z_shape.cpu())
        pred_sc = clf_sc.predict(z_shape.cpu())

        test_shapelabels = test_shapelabels.cpu().numpy()

        SSreport = accuracy_score(pred_ss,test_shapelabels)#torch.eq(test_shapelabels.cpu(), pred_ss).sum().float() / len(pred_ss)
        SCreport = accuracy_score(pred_sc,test_colorlabels.cpu().numpy())#torch.eq(test_colorlabels.cpu(), pred_sc).sum().float() / len(pred_sc)

        if confusion_mat == 1:
            cm = confusion_matrix(test_shapelabels, pred_ss)
            # Plot the confusion matrix
            plt.imshow(cm, cmap="Greys")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            for i in range(0,36):
                plt.annotate(f'{vals[i]}', (i,i), fontsize=10)
            plt.show()

    return pred_ss, pred_sc, SSreport, SCreport

#training the color map on shape and color labels
def classifier_color_train(whichdecode_use, train_dataset):
    vae.eval()
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()
        data = data.cuda()

        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,sc,j = vae(data, whichdecode_use)
        z_color = vae.sampling(mu_color, log_var_color).cuda()
        print('training color bottleneck against color labels cc')
        clf_cc.fit(z_color.cpu().numpy(), train_colorlabels)

        print('training color bottleneck against shape labels cs')
        clf_cs.fit(z_color.cpu().numpy(), train_shapelabels)

#testing the color classifier (one image at a time)
def classifier_color_test(whichdecode_use, clf_cc, clf_cs, test_dataset, verbose=0):
    vae.eval()
    with torch.no_grad():
        data, labels = next(iter(test_dataset))
        test_shapelabels=labels[0].clone()
        test_colorlabels=labels[1].clone()
        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location,sc,j = vae(data, whichdecode_use)

        z_color = vae.sampling(mu_color, log_var_color).cuda()
        pred_cc = torch.tensor(clf_cc.predict(z_color.cpu()))
        pred_cs = torch.tensor(clf_cs.predict(z_color.cpu()))

        CCreport = accuracy_score(pred_cc,test_colorlabels.cpu().numpy()) #torch.eq(test_colorlabels.cpu(), pred_cc).sum().float() / len(pred_cc)
        CSreport = accuracy_score(pred_cs,test_shapelabels.cpu().numpy()) #torch.eq(test_shapelabels.cpu(), pred_cs).sum().float() / len(pred_cs)

        if verbose==1:
            print('----**********-------color classification from color map')
            print(confusion_matrix(test_colorlabels, pred_cc))
            print(classification_report(test_colorlabels, pred_cc))

            print('----**********------shape classification from color map')
            print(confusion_matrix(test_shapelabels, pred_cs))
            print(classification_report(test_shapelabels, pred_cs))

    return pred_cc, pred_cs, CCreport, CSreport



#testing on shape for multiple images stored in memory NOT WORKING

def classifier_shapemap_test_imgs(shape, shapelabels, colorlabels,numImg, clf_shapeS, clf_shapeC, test_dataset, verbose = 0):

    global numcolors

    numImg = int(numImg)

    with torch.no_grad():
        predicted_labels=torch.zeros(1,numImg)
        shape = torch.squeeze(shape, dim=1)
        shape = shape.cuda()
        test_colorlabels = thecolorlabels(test_dataset)
        pred_ssimg = torch.tensor(clf_shapeS.predict(shape.cpu()))

        pred_scimg = torch.tensor(clf_shapeC.predict(shape.cpu()))

        SSreport = torch.eq(shapelabels.cpu(), pred_ssimg).sum().float() / len(pred_ssimg)
        SCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_scimg).sum().float() / len(pred_scimg)

        if verbose==1:
            print('----*************---------shape classification from shape map')
            print(confusion_matrix(shapelabels[0:numImg], pred_ssimg))
            print(classification_report(shapelabels[0:numImg], pred_ssimg))
            print('----************----------color classification from shape map')
            print(confusion_matrix(colorlabels[0:numImg], pred_scimg))
            print(classification_report(test_colorlabels[0:numImg], pred_scimg))
    return pred_ssimg, pred_scimg, SSreport, SCreport


#testing on color for multiple images stored in memory NOT WORKING
def classifier_colormap_test_imgs(color, shapelabels, colorlabels,numImg, clf_colorC, clf_colorS, test_dataset, verbose = 0):


    numImg = int(numImg)


    with torch.no_grad():

        color = torch.squeeze(color, dim=1)
        color = color.cuda()
        test_colorlabels = thecolorlabels(test_dataset)


        pred_ccimg = torch.tensor(clf_colorC.predict(color.cpu()))
        pred_csimg = torch.tensor(clf_colorS.predict(color.cpu()))


        CCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_ccimg).sum().float() / len(pred_ccimg)
        CSreport = torch.eq(shapelabels.cpu(), pred_csimg).sum().float() / len(pred_csimg)


        if verbose == 1:
            print('----*************---------color classification from color map')
            print(confusion_matrix(test_colorlabels[0:numImg], pred_ccimg))
            print(classification_report(colorlabels[0:numImg], pred_ccimg))
            print('----************----------shape classification from color map')
            print(confusion_matrix(shapelabels[0:numImg], pred_csimg))
            print(classification_report(shapelabels[0:numImg], pred_csimg))

        return pred_ccimg, pred_csimg, CCreport, CSreport