#%% Imports and Declarations
from utils import data_encoder_pytorch
import pandas as pd
import cv2, time
import torch
from tqdm import tqdm
import numpy as np
from Pytorch_Trainer import pytorch_trainer
from sklearn.model_selection import train_test_split

DATASET_PATH = '/media/piyush/A33B-9070/Object_detection/Vision/Cats_dog_labeled_box/train.csv'
GRID_DIM = (7,7)
B = 1 # Anchors
#%% Load Dataset
dataset = pd.read_csv(DATASET_PATH)
dataset = dataset[:10]
print(dataset.info())
sample_count = dataset.shape[0]
images = []
image_sizes = []

# Read Images
for path in tqdm(dataset['path'].values):
    image = cv2.imread(path)
    image_sizes.append([image.shape[0], image.shape[1]])
    image = cv2.resize(image, (224, 224))
    images.append(image)
images = np.array(images)
dimentions = dataset.values[:, 3:7]

print("Number of Images:", images.shape[0])
# Convert to torch Format
dimentions_torch = torch.from_numpy(dimentions.astype(np.float32))
image_sizes_torch = torch.FloatTensor(image_sizes)

#%%Generate Grid for each image
encoder = data_encoder_pytorch(GRID_DIM)
Xgrid, Ygrid, Xrel_c, Yrel_c, Xrel_w, Yrel_w = encoder.annotaion_encoding(dimentions_torch.clone(), image_sizes_torch)

grid_coords = torch.hstack( (Xgrid.unsqueeze(1), Ygrid.unsqueeze(1)) )
rel_center  = torch.hstack( (Xrel_c.unsqueeze(1), Yrel_c.unsqueeze(1)) )
rel_width   = torch.hstack( (Xrel_w.unsqueeze(1), Yrel_w.unsqueeze(1)) )
classes     = torch.FloatTensor([0 for _ in range(sample_count)])
class_count = 1

grids = encoder.to_grids(grid_coords, rel_center, rel_center, classes, class_count)
print(grids.shape)

#%% Split Data
X = images.reshape((-1, 3, 224, 224))
y = grids.detach().cpu().numpy()

y = y.reshape( (-1, GRID_DIM[0]*GRID_DIM[1]*(class_count + B*5)) )

print(type(X))
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test  = torch.FloatTensor(y_test)

print(X_train.shape)
print(y_train.shape)
# %% load model
from model import yolov1
inchannels = 3
class_count = 1
model = yolov1(inchannels, GRID_DIM, class_count)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)
trainer = pytorch_trainer(model, criteria=None, optimizer=optimizer)
trainer.summary(input_shape=(3, 224, 224))

#%% Test Loss Function
class FocalLoss(torch.nn.Module):
    def __init__(self, grid_dim, class_count, anchor_count=1):
        super(FocalLoss, self).__init__()
        self.S = grid_dim
        self.B = anchor_count
        self.C = class_count

        self.lambda_obj  = 5
        self.lambda_noobj= 0.5
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, prediction, actual):
        shape = (-1, self.S[0], self.S[1], self.C + self.B*5)
        prediction_reshaped = prediction.view(shape)
        actual_reshaped     = actual.view(shape)

        actual_presence = actual_reshaped[..., self.C + 0].unsqueeze(3)  # S, S, 1
        
        '''
        print(actual_presence.shape)
        
        actual_presence_ = actual_reshaped[..., self.C + 0] == 1.0
        print(actual_presence_)
        print(prediction_reshaped[actual_presence_])
        
        # ==========REGRESSION LOSS============
        # [c1...cn, p, x, y, w, h]
        
        pred_x   = prediction_reshaped[actual_presence_][:, self.C + 1]
        pred_y   = prediction_reshaped[actual_presence_][:, self.C + 2]
        actual_x = actual_reshaped[actual_presence_][:, self.C + 1]
        actual_y = actual_reshaped[actual_presence_][:, self.C + 2]

        diff_squared = torch.square(actual_x - pred_x) + torch.square(actual_y - pred_y)
        regression_loss = torch.sum(diff_squared)
        
        print("AX:", actual_x)
        print("AY:", actual_y)
        print("PX:", pred_x)
        print("PY:", pred_y)
        print(diff_squared)
        print("LOSS:", regression_loss)
        '''
        
        # ==========REGRESSION LOSS============
        # [c1...cn, p, x, y, w, h]
        regression_loss = self.mse( actual_presence * actual_reshaped[..., self.C+1 : self.C + 3],
                                    actual_presence * prediction_reshaped[..., self.C+1 : self.C + 3])
        regression_loss += self.mse( actual_presence * torch.sqrt(actual_reshaped[..., self.C + 3: self.C + 5]),
                                     actual_presence * torch.sqrt(prediction_reshaped[..., self.C + 3: self.C + 5]) )        
        regression_loss = self.lambda_obj * regression_loss
        print("Regression Loss:", regression_loss)

        #===========CONFIDENCE LOSS============
        confidence_loss  = self.mse( actual_presence * actual_reshaped[..., 0:self.C], 
                                     actual_presence * prediction_reshaped[..., 0:self.C] )
        _noobj_conf      = self.mse( (1 - actual_presence) * actual_reshaped[..., 0:self.C], 
                                     (1 - actual_presence) * prediction_reshaped[..., 0:self.C] )
        confidence_loss += self.lambda_noobj * _noobj_conf
        
        print("Confidence Loss:", confidence_loss)

    


        
loss = FocalLoss(GRID_DIM, class_count, anchor_count=1)
loss(y_train[2:4], y_train[0:2])

# %%
