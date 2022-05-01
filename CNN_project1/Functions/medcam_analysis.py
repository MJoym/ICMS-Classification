"""Function that implements Grad-CAM using med-cam"""

import torch
from medcam import medcam
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import cv2 as cv2
import os
from os import path
import matplotlib.pyplot as plt

date_cond = 'leg_2009_01_29'
session = 'f'
trial = '1002'
medcam_backend = 'gcampp'

background_img = trial+'_avg_bnw.png'
saved_img_title = date_cond+'_'+session+'_'+trial+'_'+medcam_backend+'2.jpg'
path_of_bckg_img = '/media/joy/Elements/Joanna/ICMS/Legolas/'+date_cond+'/'+session+'/devidedBlank/'+background_img
create_new_dir = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Grad Cam Pictures/medcam/new/map_'+date_cond+'_'+session+'_'+trial
filepathSaveImg = create_new_dir+'/'+saved_img_title


def interpolate_heatmap(img_path, original_heatmap):
    img = cv2.imread(img_path)
    original_heatmap_np = original_heatmap.numpy()
    heatmap = cv2.resize(original_heatmap_np, (img.shape[1], img.shape[0]))
    # plt.matshow(heatmap)
    # plt.show()
    heatmap = np.uint8(heatmap)
    # plt.imshow(heatmap)
    # plt.show()
    # print(type(heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + img * 0.7
    cv2.imwrite(filepathSaveImg, superimposed_img)


def medcam_analysis(model, test_dataloader):
    medcam_model = medcam.inject(model, output_dir=create_new_dir, layer='auto', backend=medcam_backend, save_maps=True)
    print(medcam.get_layers(medcam_model))
    for (x,y) in test_dataloader:
        x = x[:, None, :, :, :]
        x = x.type(torch.FloatTensor)
        # make the predictions and calculate the validation loss
        pred_test = medcam_model(x)
        test_loss = F.cross_entropy(pred_test, y)
        # calculate the number of correct predictions
        pred_t = pred_test.argmax(1)

        # filep = '/home/joy/Documents/Neuroscience_Master/Neural_Networks/CNN_project1/Functions/attention_maps/2503_c_2009/max3/attention_map_0_0_0.nii'
        if path.isdir(create_new_dir):
            file_path = create_new_dir+'/max3/attention_map_0_0_0.nii.gz'

        else:
            os.mkdir(create_new_dir)
            file_path = create_new_dir+'/max3/attention_map_0_0_0.nii.gz'

        test_load = nib.load(file_path).get_fdata()
        print(test_load.shape)

        # average the channels of the activations
        heatmap = torch.from_numpy(test_load)
        heatmap = torch.mean(heatmap, dim=[2]).squeeze()
      
        interpolate_heatmap(path_of_bckg_img, heatmap)
