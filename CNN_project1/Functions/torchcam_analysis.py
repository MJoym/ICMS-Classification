import torch
from medcam import medcam
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import cv2 as cv2
import os
from os import path
from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp
import matplotlib.pyplot as plt


date_cond = 'leg_2009_03_25'
session = 'd'
trial = '6006'
torchcam_backend = 'torchcam_gcampp'

background_img = trial+'_avg_bnw.png'
saved_img_title = trial+'_'+torchcam_backend+'_SGD_max2.jpg'
path_of_bkg_img = '/media/joy/Elements/Joanna/ICMS/Legolas/'+date_cond+'/'+session+'/devidedBlank/'+background_img
create_new_dir = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Grad Cam Pictures/torchcam/SGD_Model_45/'+session+'_'+date_cond+'/'
filepathSaveImg = create_new_dir+'/'+saved_img_title


def interpolate_heatmap(img_path, original_heatmap):
    img = cv2.imread(img_path)
    original_heatmap_np = original_heatmap.numpy()
    heatmap = cv2.resize(original_heatmap_np, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    # plt.imshow(heatmap)
    # plt.show()
    print(type(heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + img * 0.7

    if path.isdir(create_new_dir):
        file2SaveImg = create_new_dir+'/'+saved_img_title

    else:
        os.makedirs(create_new_dir)
        file2SaveImg = create_new_dir+'/'+saved_img_title

    cv2.imwrite(file2SaveImg, superimposed_img)


def torchcam_analysis(model, test_dataloader):
    # Torchvision GradCAM, GradCAM++
    cam_extractor = GradCAMpp(model, target_layer='max2')

    for (x,y) in test_dataloader:
        x = x[:, None, :, :, :]
        x = x.type(torch.FloatTensor)

        # make the predictions and calculate the validation loss
        out = F.sigmoid(model(x))
        test_loss = F.cross_entropy(out, y)
        # calculate the number of correct predictions
        pred_t = out.argmax(1)

        # Retrieve the CAM by passing the class index and the model output
        checksquezeeout = out.squeeze(0)
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        activation_map = activation_map[0]

        # average the channels of the activations
        heatmap = torch.mean(activation_map, dim=[0]).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        # heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        # heatmap /= torch.max(heatmap)

        # Visualize the raw CAM
        # plt.matshow(heatmap)

        # # Visualize the raw CAM
        # plt.imshow(activation_map[0].numpy())
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        # Interpolation of image and heatmap:
        imgPath = path_of_bkg_img
        interpolate_heatmap(imgPath, heatmap)
