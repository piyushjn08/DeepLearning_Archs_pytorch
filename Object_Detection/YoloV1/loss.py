#%% Imports and Declarations
import torch
import torch.nn as nn

'''
Not designed for multiple boxes yet
'''
class FocalLoss(nn.Module):
    def __init__(self, grid_dim, class_count, B=1):
        self.grid_dim    = grid_dim
        self.box_count   = B
        self.class_count = class_count

        self.lambda_obj  = 5
        self.lambda_noobj= 0.5
    
    def forward(self, prediction, actual):
        shape = (-1, self.grid_dim[0], self.grid_dim[1], self.class_count + self.box_count*5)
        prediction_reshaped = torch.reshape(prediction, shape)
        actual_reshaped     = torch.reshape(actual, shape)

        prediction_presence = torch.where(actual_reshaped[..., self.class_count + 0] == 1.0)
        print(prediction_presence)


