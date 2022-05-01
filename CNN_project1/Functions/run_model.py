import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from classes.ICMSet import ICMSet
from Functions.test_data import test
from Functions.plt_loss_accuracy import plot_pred, plot_sample
from Functions.GradCAM3D import GradCam
from Functions.medcam_analysis import medcam_analysis
from Functions.torchcam_analysis import torchcam_analysis
from torch.utils.data.dataset import random_split
from medcam import medcam
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt

############################################# GRADCAM ##########################################################

# from Functions.V1_3DCNN_GRADCAM import V1_3DCNN

# test_batch_size = 1
# # define a 1 image dataset
# dataset_dir = "/media/joy/Elements/Joanna/ICMS/CNN/GradCam_Labels"
#
# # Load the train and test data:
# dataset = ICMSet(dataset_dir, transform=transforms.ToTensor())
# test_dl = DataLoader(dataset, test_batch_size, num_workers=0, shuffle=False, pin_memory=False)
# print("\nthe_model with GradCAM:")
# pathf = '/home/joy/Documents/Neuroscience_Master/Neural_Networks/CNN_project1/Modelsaved/ADAM_Model/the_model_60.pt'
# the_model = V1_3DCNN(numChannels=1, numLabels=2)
# the_model.load_state_dict(torch.load(pathf))
# the_model.eval()

# GradCam(the_model, test_dl)
# torchcam_analysis(the_model,test_dl)

# medcam_analysis(the_model, test_dl)

############################################# GRADCAM ##########################################################


############################################ TESTING ##########################################################
#
from classes.V1_3DCNN import V1_3DCNN

# Seed
seed = 5
torch.manual_seed(seed)

# test_split = .999  # percentage of total data to use for training
# train_split = .001
# batch_size = 4
train_split = .75  # percentage of total data to use for training
val_split = .25
batch_size = 8
test_batch_size = 32
# epochs = 25  # Total epochs to run the training

# Train and test data directory
dataset_dir = "/media/joy/Elements/Joanna/ICMS/CNN/Labels"

# Load the train and test data:
dataset = ICMSet(dataset_dir, transform=transforms.ToTensor())

# # Visualize the data for debugging
# img, label = dataset[0]
# print(img.shape, label)
# plot_sample(img)

# Split the data into train, test and validation:
# test_size = int(len(dataset)*test_split)
# val_size = len(dataset) - test_size
# test_data, val_data = random_split(dataset,[test_size,val_size])

# Split the data into train, test and validation:
train_size = int(len(dataset)*train_split)
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset,[train_size,test_size])
val_size = int(val_split*train_size)
train_size = train_size - val_size
train_data, val_data = random_split(train_data, [train_size, val_size])

print(f"Length of Data : {len(dataset)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Test Data : {len(test_data)}")

test_dl = DataLoader(test_data, test_batch_size, shuffle=False, num_workers=0, pin_memory=False)
print("\nthe_model saved (TESTING):")

pathf = '/home/joy/Documents/Neuroscience_Master/Neural_Networks/CNN_project1/Model saved/ADAM_Model/the_model_60.pt'
the_model = V1_3DCNN(numChannels=1, numLabels=2)
the_model.load_state_dict(torch.load(pathf))
the_model.eval()

T = test(the_model, test_dl)
############################################# TESTING ##########################################################
