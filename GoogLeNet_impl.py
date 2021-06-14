#%% Imports and declarations
import torch
import torch.nn as nn 
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from random import randrange
from Pytorch_Trainer import pytorch_trainer

class conv_layer(nn.Module):
    def __init__(self, inchannels, out_channels, **kwargs):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(inchannels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
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

class GoogleNet(nn.Module):
    def __init__(self, inchannels, outclasses):
        super(GoogleNet, self).__init__()
        self.conv1 = conv_layer(inchannels, 64, kernel_size=7, stride=2, padding=3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_layer(64, 192, kernel_size=3, stride=1, padding=1)
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1*1*1024, outclasses)
    
    def forward(self, batch):
        x = self.conv1(batch)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.maxp(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxp(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxp(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        output1 = self.linear(x)

        return output1





#%% Test Model
classes = 10
set_count = 10
X = torch.randn(set_count, 3, 224, 224)
y = []
for i in range(set_count):
    index = randrange(classes) # Select random index from class count
    y.append(index)
y = torch.LongTensor(y)

print("Batch Shape:", X.shape, flush=True)

# Define Model
model = GoogleNet(3, classes)

# Define Tuning Parameters
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Initialize TrainerClass
trainer = pytorch_trainer(model, criteria, optimizer)
trainer.summary(X)

# Train Model
trainer.fit(X, y, epochs=50, batch_size=4, calculate_acc=True)
# %% train model

