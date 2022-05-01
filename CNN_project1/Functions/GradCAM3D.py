""" GradCam implementation by J.M.F. based on Medium article by Stepan Ulyanin "Implementing Grad-CAM in PyTorch".
Works only with one sample at a time. Please use folder "GradCAM Label" for this script."""
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv2
import os
from os import path


date_cond = 'leg_2009_01_29'
session = 'f'
trial = '3006'

background_img = trial+'_avg_bnw.png'
saved_img_title = date_cond+'_'+session+'_'+trial+'_SGD45.jpg'
filepathSaveImg = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Grad Cam Pictures/SGD_Model_45_MyGRADCAM/'+session+'_'+date_cond+'/'
path_of_bkg_img = '/media/joy/Elements/Joanna/ICMS/Legolas/'+date_cond+'/'+session+'/devidedBlank/'+background_img


def interpolate_heatmap(img_path, original_heatmap):
    img = cv2.imread(img_path)
    original_heatmap_np = original_heatmap.numpy()
    heatmap = cv2.resize(original_heatmap_np, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.8 + img * 0.7

    if path.isdir(filepathSaveImg):
        file2SaveImg = filepathSaveImg+saved_img_title

    else:
        os.mkdir(filepathSaveImg)
        file2SaveImg = filepathSaveImg+saved_img_title

    cv2.imwrite(file2SaveImg, superimposed_img)


def GradCam(model, test_dl):
    for (x, y) in test_dl:
            x = x[:, None, :, :, :]
            x = x.type(torch.FloatTensor)
            # make the predictions and calculate the validation loss
            pred_test = F.sigmoid(model(x))
            test_loss = F.cross_entropy(pred_test, y)
            # calculate the number of correct predictions
            pred_t = pred_test.argmax(1)

            one_hot = np.zeros((1, pred_test.size()[-1]), dtype=np.float32)
            correct_label = y.numpy().tolist()
            one_hot[0][correct_label] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot*pred_test)
            # get the gradient of the output with respect to the parameters of the model
            model.zero_grad()
            one_hot.backward(retain_graph=True)

            # pull the gradients out of the model
            gradients = model.get_activation_gradients()

            # pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2])
            pooled_gradients2 = torch.mean(gradients, dim=[0])

            # get the activations of the last convolutional layer
            activations = model.get_activations(x).detach()

            # weight the channels by corresponding gradients
            depth = 256
            for i in range(depth):
                activations[:, i, :, :, :] *= pooled_gradients2[i]

            # average the channels of the activations
            heatmap = torch.mean(activations, dim=[1,2]).squeeze()

            # relu on top of the heatmap
            # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
            heatmap = np.maximum(heatmap, 0)

            # normalize the heatmap
            heatmap /= torch.max(heatmap)

            # Interpolation of image and heatmap:
            imgPath = path_of_bkg_img
            interpolate_heatmap(imgPath, heatmap)
