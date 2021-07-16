#%% Imports and Declarations
from model import yolov1
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import numpy as np
from utils import data_encoder_pytorch
import warnings

inchannels = 3
class_count = 1
GRID_DIM = (7, 7)

DATASET_PATH = '/media/crl/A33B-9070/Object_detection/Vision/Cats_dog_labeled_box/train.csv'

def warn(*args, **kwargs):
    pass
warnings.warn = warn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
#%% Load Dataset
dataset = pd.read_csv(DATASET_PATH)
dataset = dataset[:10]
sample_count = dataset.shape[0]
images = []
images_original = []
image_sizes = []

# Read Images
for path in tqdm(dataset['path'].values):
    image = cv2.imread(path)
    image_sizes.append([image.shape[0], image.shape[1]])
    images_original.append(image)
    
    image = cv2.resize(image, (224, 224))
    images.append(image)

images = np.array(images)
dimentions = dataset.values[:, 3:7]

print("Number of Images:", images.shape[0])
# Convert to torch Format
images_torch = torch.FloatTensor(images).reshape((-1, 3, images.shape[1], images.shape[2])).to(device)
image_sizes_torch = torch.FloatTensor(image_sizes).to(device)
GRID_DIM_torch = torch.LongTensor(GRID_DIM).to(device)

images = images / 255.0

#%% Load model
print("Loading Model...")
model_dict = torch.load('/home/crl/Music/Misc/checkpoints/99-1.07.pth.tar')
model = yolov1(inchannels, GRID_DIM, class_count)

model.load_state_dict(model_dict['state_dict'])
model = model.to(device)
print(model.parameters())

#%% Make predictions
import time
start = time.time()
decoder = data_encoder_pytorch(GRID_DIM)

for index in range(images_torch.shape[0]):
    output = model.forward(images_torch[index:index+1])
    print()
    grid_coords, rel_center, rel_width = decoder.find_box(output, GRID_DIM, class_count, boxes=1, conf_threshold=0.8)
    bbox = decoder.decode_grid(image_sizes_torch[index:index+1], grid_coords, rel_center, rel_width)
    print("Image Size:", image_sizes[index])
    print("BOX Size:", bbox[0].tolist())
    print("Grid Coord:", grid_coords.tolist())
    
    
    key = decoder.show_image(bbox.tolist()[0], images_original[index])
    if(key == ord('q')):
        break
    
print("Time Taken", time.time() - start)
# %%
