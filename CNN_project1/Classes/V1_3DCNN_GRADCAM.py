import torch.nn as nn
from torch.nn import Module, Conv3d, Linear, MaxPool3d, ReLU, Flatten, AvgPool3d, Sigmoid, GELU, Hardswish
import warnings
warnings.filterwarnings('ignore')


class V1_3DCNN(Module):
    def __init__(self, numChannels=1, numLabels=2):
        # Call the parent constructor:
        super(V1_3DCNN, self).__init__()

        # CNN Block1
        self.bn1 = nn.BatchNorm3d(1)
        self.conv1a = Conv3d(in_channels=numChannels, out_channels=32, kernel_size=(3,3,3), stride = (1,1,1))
        self.conv1a_bn = nn.BatchNorm3d(32)
        self.conv1b = Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))
        self.conv1b_bn = nn.BatchNorm3d(64)
        self.max1 = MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        # CNN Block2
        self.conv2a = Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))
        self.conv2a_bn = nn.BatchNorm3d(64)
        self.conv2b = Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1))
        self.conv2b_bn = nn.BatchNorm3d(128)
        self.max2 = MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        # CNN Block3
        self.conv3a = Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1))
        self.conv3a_bn = nn.BatchNorm3d(128)
        self.conv3b = Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1))
        self.conv3b_bn = nn.BatchNorm3d(256)
        self.max3 = MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        # Fully Connected layer
        self.flat = Flatten()
        self.fc1 = Linear(256 * 9 * 9 * 2, 128)
        self.fc2 = Linear(128, 50)
        self.fc3 = Linear(50, numLabels)

        self.relu = ReLU()
        self.dropout = nn.Dropout3d(p=0.2)
        # For Grad-CAM
        self.gradients = None

    def activation_hook(self, grad):
        self.gradients = grad

    # Method for the gradients' extraction
    def get_activation_gradients(self):
        return self.gradients

    # Method for the activations' extraction
    def get_activations(self, x):
        return self.forward_conv(x)

    def forward_conv(self, x):
        output = self.bn1(x)
        # Block1:
        output = self.relu(self.conv1a_bn(self.conv1a(output)))
        output = self.relu(self.conv1b_bn(self.conv1b(output)))
        output = self.max1(output)

        # Block2:
        output = self.relu(self.conv2a_bn(self.conv2a(output)))
        output = self.relu(self.conv2b_bn(self.conv2b(output)))
        output = self.max2(output)

        # Block3:
        output = self.relu(self.conv3a_bn(self.conv3a(output)))
        output = self.relu(self.conv3b_bn(self.conv3b(output)))
        output = self.max3(output)

        return output

    def forward_fc(self, x):
        # Fully connected layers:
        output = self.flat(x)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)  # BCELoss needs sigmoid at the last

        return output

    def forward(self, x):
        output = self.forward_conv(x)
        # # For Grad-CAM:
        h = output.register_hook(self.activation_hook)
        output = self.forward_fc(output)
        return output

