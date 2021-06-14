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

class inception_block(nn.Module):
    def __init__(self, inchannels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inception_block, self).__init__()
        self.output1 = conv_layer(inchannels, out_1x1, kernel_size=1, stride=1, padding=0)
        
        self.output2 = nn.Sequential(conv_layer(inchannels, red_3x3, kernel_size=1, stride=1, padding=0),
                                    conv_layer(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1))
        
        self.output3 = nn.Sequential(conv_layer(inchannels, red_5x5, kernel_size=1, stride=1, padding=0),
                                    conv_layer(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2))
        
        self.output4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    conv_layer(inchannels, out_1x1pool, kernel_size=1, stride=1, padding=0))
        
    def forward(self, data):
        return torch.cat([self.output1(data), self.output2(data), self.output3(data), self.output4(data)], 1)