#%% Imports and Declarations
import torch
import torch.nn as nn

'''
Not designed for multiple boxes yet
'''
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
        # ==========REGRESSION LOSS============
        # [c1...cn, p, x, y, w, h]
        # Location of BOX
        print()
        regression_loss = self.mse( actual_presence * actual_reshaped[..., self.C+1 : self.C + 3],
                                    actual_presence * prediction_reshaped[..., self.C+1 : self.C + 3])
        print("R1", regression_loss)
        sqrt_actual = torch.sign(actual_reshaped[..., self.C + 3: self.C + 5]) * torch.sqrt(
                        torch.abs(actual_reshaped[..., self.C + 3: self.C + 5]) + 1e-6 )

        sqrt_preds  = torch.sign(prediction_reshaped[..., self.C + 3: self.C + 5]) * torch.sqrt(
                        torch.abs(prediction_reshaped[..., self.C + 3: self.C + 5]) + 1e-6)
        regression_loss += self.mse( actual_presence * sqrt_actual, actual_presence * sqrt_preds)
        print("R2", regression_loss)
        regression_loss = self.lambda_obj * regression_loss
        print("Regression Loss:", regression_loss.item())

        #===========CONFIDENCE LOSS============
        # If an object is detected in the box, the confidence loss (measuring the objectness of the box)
        confidence_loss  = self.mse( actual_presence * actual_reshaped[..., self.C:self.C+1], 
                                     actual_presence * prediction_reshaped[..., self.C:self.C+1] )
        _noobj_conf      = self.mse( (1 - actual_presence) * actual_reshaped[..., self.C:self.C+1], 
                                     (1 - actual_presence) * prediction_reshaped[..., self.C:self.C+1] )
        confidence_loss += self.lambda_noobj * _noobj_conf
        
        print("Confidence Loss:", confidence_loss.item())

        #=========Classification Loss==========
        # Class of BOX
        classification_loss = self.mse( actual_presence * actual_reshaped[..., 0:self.C],
                                        actual_presence * prediction_reshaped[..., 0:self.C])
    
        print("Classification Loss:", classification_loss.item())

        loss = regression_loss + confidence_loss + classification_loss
        print("LOSS:", loss)
        return loss


