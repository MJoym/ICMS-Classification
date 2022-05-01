import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from classes.V1_3DCNN import V1_3DCNN
import time
from classes.ICMSet import ICMSet
from torchsummary import summary
from torch.utils.data.dataset import random_split
from Functions.train_data import train
from Functions.init_weigths import initialize_weights
from Functions.plt_loss_accuracy import plot_pred, plot_sample
from Functions.test_data import test
from torch.optim import Adam, SGD
import pickle

# Seed
seed = 5
torch.manual_seed(seed)

train_split = .8  # percentage of total data to use for training
val_split = .15
batch_size = 8
test_batch_size = 32
epochs = 100 # Total epochs to run the training

# Train and test data directory
dataset_dir = "/media/joy/Elements/Joanna/ICMS/python scripts for neural network course/CNN/Labels"

# Load the train and test data:
dataset = ICMSet(dataset_dir, transform=transforms.ToTensor())

# Visualize the data for debugging
img, label = dataset[0]
print(img.shape, label)
plot_sample(img)

# Split the data into train, test and validation:
train_size = int(len(dataset)*train_split)
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset,[train_size,test_size])
val_size = int(val_split*train_size)
train_size = train_size - val_size
train_data, val_data = random_split(train_data, [train_size, val_size])

print(f"Length of Data : {len(dataset)}")
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Test Data : {len(test_data)}")


# Creating the batches with Dataloader to speed up the code:
train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_dl = DataLoader(val_data, test_batch_size, num_workers=0, pin_memory=False)
test_dl = DataLoader(test_data, test_batch_size, num_workers=0, pin_memory=False)

# # Initialize our model:
model = V1_3DCNN(numChannels=1, numLabels=2)

# filepath = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/the_model_5.pt'
# model.load_state_dict(torch.load(filepath))
model.apply(initialize_weights)
epoch_start = 0

print(summary(model, (1,50,100,100)))
#etas = [0.0001, 0.0005, 0.002, 0.01, 0.05, 0.1]
etas = [0.0001]
lossFn = nn.CrossEntropyLoss()

# Time how long the training takes:
startTime = time.time()
filepath = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/ADAM_Model/the_model_60_checkp.pt'

# Loop over gradient steps (eta):
for eta in etas:

    # opt = SGD(model.parameters(), eta)
    opt = SGD(model.parameters(), eta, weight_decay=0.01)
    print("Training and Validation when learning rate =", eta)

    H = train(model, train_dl, val_dl, epochs, epoch_start, lossFn, opt)
    # finish measuring how long training took:

    endTime = time.time()
    totalTime = (endTime - startTime) / 60
    print("[INFO] total time taken to train the model: {:.2f}m".format(totalTime))

    # Save the final model
    torch.save(model.state_dict(), filepath)
    plot_pred(H)

    # Save training history:
    pickle.dump(H, open("TrainingHistoryVar_SGD.dat", "wb"))

    # Test the model:
    T = test(model, test_dl)
