#%% Imports and Declarations
import torch
import torch.nn as nn
from Pytorch_Trainer import pytorch_trainer
from random import randrange

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
    def  __init__(self, inchannels, out_1x1, out_3x3, out2_1x1, stride=1):
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
            # Sequential without Relu
            identity_converter = nn.Sequential(nn.Conv2d(self.inchannels, self.out2_1x1, kernel_size=1, stride=self.stride, padding=0),
                                                nn.BatchNorm2d(self.out2_1x1))
            identity = identity_converter(identity)
        
        x = x + identity
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, inchannels, outclasses):
        super(ResNet50, self).__init__()
        self.conv1   = nn.Conv2d(inchannels, 64, kernel_size=7, stride=2, padding=3) # Size Reduction
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Size Reduction
        self.conv2_1 = residual_block(64, 64, 64, 256, stride=1)
        self.conv2_2 = residual_block(256, 64, 64, 256, stride=1)
        self.conv2_3 = residual_block(256, 64, 64, 256, stride=1)

        self.conv3_1 = residual_block(256, 128, 128, 512, stride=2) # Size Reduction
        self.conv3_2 = residual_block(512, 128, 128, 512, stride=1)
        self.conv3_3 = residual_block(512, 128, 128, 512, stride=1)
        self.conv3_4 = residual_block(512, 128, 128, 512, stride=1)

        self.conv4_1 = residual_block(512, 256, 256, 1024, stride=2) # Size Reduction
        self.conv4_2 = residual_block(1024, 256, 256, 1024, stride=1)
        self.conv4_3 = residual_block(1024, 256, 256, 1024, stride=1)
        self.conv4_4 = residual_block(1024, 256, 256, 1024, stride=1)
        self.conv4_5 = residual_block(1024, 256, 256, 1024, stride=1)
        self.conv4_6 = residual_block(1024, 256, 256, 1024, stride=1)

        self.conv5_1 = residual_block(1024, 512, 512, 2048, stride=2) # Size Reduction
        self.conv5_2 = residual_block(2048, 512, 512, 2048, stride=1)
        self.conv5_3 = residual_block(2048, 512, 512, 2048, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # output size of 1x1x2048
        self.linear = nn.Linear(2048, outclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.linear(x)

        return x

#%% Create Sample Data
sample_size = 10
class_count = 10

X = torch.randn((sample_size, 3, 224, 224))
y = []
for i in range(sample_size):
    y.append(randrange(class_count))

X = torch.FloatTensor(X)
y = torch.LongTensor(y)

print(X.shape)
print(y.shape)
# %% Test Model
model = ResNet50(3, class_count)

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

trainer = pytorch_trainer(model, criteria=criteria, optimizer=optimizer)
print(trainer.summary(X.shape[1:]))
#trainer.trace_model(input_shape=X.shape[1:])
trainer.fit(X, y, epochs=10, calculate_acc=True)


# %%
