import torch
import torch.nn as nn 

class conv_layer(nn.Module):
    def __init__(self, inchannels, out_channels, **kwargs):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(inchannels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x = self.conv(data)
        x = self.bn(x)
        x = self.relu(x)
        return x

class yolov1(nn.Module):
    def __init__(self, inchannels, grid_dim, class_count):
        # input image size : 224, 224
        super(yolov1, self).__init__()
        self.conv1 = conv_layer(inchannels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = conv_layer(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxp  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = conv_layer(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = conv_layer(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = conv_layer(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv6 = conv_layer(256, 512, kernel_size=3, stride=1, padding=1)

        self.conv7 = conv_layer(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv8 = conv_layer(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv9 = conv_layer(512, 1024, kernel_size=3, stride=1, padding=1)

        self.conv10 = conv_layer(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv11 = conv_layer(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv12 = conv_layer(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.conv13 = conv_layer(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv14 = conv_layer(1024, 1024, kernel_size=3, stride=1, padding=1)
    
        self.fc1 = nn.Linear(7*7*1024, 496)
        self.fc2 = nn.Linear(496, grid_dim*grid_dim*(class_count + 5))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxp(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxp(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.maxp(x)

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
