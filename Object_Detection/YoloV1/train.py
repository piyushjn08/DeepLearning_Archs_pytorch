#%% Imports and Declarations
from utils import data_encoder_pytorch
import pandas as pd
import cv2
import torch
from tqdm import tqdm
import numpy as np

DATASET_PATH = '/media/piyush/A33B-9070/Object_detection/Vision/Cats_dog_labeled_box/train.csv'
#%% Load Dataset
dataset = pd.read_csv(DATASET_PATH)
print(dataset.info())

encoder = data_encoder_pytorch((7,7))
images = []
image_sizes = []
dimentions = dataset.values[:, 3:7]

for path in tqdm(dataset['path'].values):
    image = cv2.imread(path)
    image_sizes.append([image.shape[0], image.shape[1]])
    image = cv2.resize(image, (224, 224))
    images.append(image)
images = np.array(images)
dimentions = encoder.resize_bb_coord(image_sizes, (224, 224), dimentions)

print("Number of Images:", images.shape[0])
for i in range(5):
    encoder.show_image(dimentions[i], images[i])

