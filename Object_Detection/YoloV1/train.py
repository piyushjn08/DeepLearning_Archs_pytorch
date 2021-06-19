#%% Imports and Declarations
from utils import data_encoder_pytorch
import pandas as pd
import cv2, time
import torch
from tqdm import tqdm
import numpy as np
from Pytorch_Trainer import pytorch_trainer

DATASET_PATH = '/media/piyush/A33B-9070/Object_detection/Vision/Cats_dog_labeled_box/train.csv'
#%% Load Dataset
dataset = pd.read_csv(DATASET_PATH)
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
encoder = data_encoder_pytorch((7,7))
Xgrid, Ygrid, Xrel_c, Yrel_c, Xrel_w, Yrel_w = encoder.annotaion_encoding(dimentions_torch.clone(), image_sizes_torch)

grid_coords = torch.hstack( (Xgrid.unsqueeze(1), Ygrid.unsqueeze(1)) )
rel_center  = torch.hstack( (Xrel_c.unsqueeze(1), Yrel_c.unsqueeze(1)) )
rel_width   = torch.hstack( (Xrel_w.unsqueeze(1), Yrel_w.unsqueeze(1)) )
classes     = torch.FloatTensor([0 for _ in range(sample_count)])
class_count = 1

grids = encoder.to_grids(grid_coords, rel_center, rel_center, classes, class_count)
print(grids.shape)
# %% load model
from model import yolov1
inchannels = 3
model = yolov1(inchannels, 12)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)
trainer = pytorch_trainer(model, criteria=None, optimizer=optimizer)
trainer.summary(input_shape=(3, 224, 224))

# %%
