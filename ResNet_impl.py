#%% Imports and Declarations
import torch
import torch.nn as nn
from Pytorch_Trainer import pytorch_trainer

class conv_layer(nn.Module):
    def __init__(self, inchannels, outchannels, **kwargs):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, **kwargs)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Ix = (I - K + 2P)/S + 1
class residual_block(nn.Module):
    def  __init__(self, inchannels, out_1x1, out_3x3, out2_1x1, identity_downsample = None, stride=1):
        super(residual_block, self).__init__()
        self.stride = stride
        self.inchannels = inchannels
        self.out2_1x1 = out2_1x1

        self.conv1 = conv_layer(inchannels, out_1x1, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_layer(out_1x1, out_3x3, kernel_size=3, stride=stride, padding=1)
        self.conv3 = conv_layer(out_3x3, out2_1x1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
                
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # if input shape and channels does not match with output shape and channels,
        # then reshape input using conv layer
        
        if self.stride!=1 or self.inchannels != self.out2_1x1:
            identity = conv_layer(self.inchannels, self.out2_1x1, kernel_size=1, stride=1, padding=0)
        
        x = x + identity
        x = self.relu(x)
        return x

#%% 