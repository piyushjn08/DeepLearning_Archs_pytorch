#%% Imports and Declarations
from utils import data_encoder_pytorch
import pandas as pd
import cv2, time
import torch
from tqdm import tqdm
import numpy as np
from Pytorch_Trainer import pytorch_trainer
from sklearn.model_selection import train_test_split
from loss import FocalLoss, YoloLoss

DATASET_PATH = '/media/crl/A33B-9070/Object_detection/Vision/Cats_dog_labeled_box/train.csv'
GRID_DIM = (7,7)
B = 1 # Anchors
#%% Load Dataset
dataset = pd.read_csv(DATASET_PATH)
dataset = dataset[:100]
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
X = X / 255.0
y = grids.detach().cpu().numpy()

y = y.reshape( (-1, GRID_DIM[0]*GRID_DIM[1]*(class_count + B*5)) )

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

model     = yolov1(inchannels, GRID_DIM, class_count).to('cuda')
criteria  = FocalLoss(GRID_DIM, class_count, anchor_count=1)
#criteria = YoloLoss(GRID_DIM, class_count, B=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
trainer = pytorch_trainer(model, criteria=criteria, optimizer=optimizer, lr_patience=15)
#trainer.summary(input_shape=(3, 224, 224))

#%% Train Model
trainer.fit(X_train, y_train, batch_size=8,epochs=1000, anomaly_detection=True)

# %%
