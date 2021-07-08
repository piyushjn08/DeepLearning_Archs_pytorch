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
        regression_loss = self.mse( torch.flatten(torch.multiply(actual_presence, actual_reshaped[..., self.C+1 : self.C + 3])),
                                    torch.flatten(torch.multiply(actual_presence, prediction_reshaped[..., self.C+1 : self.C + 3])) )
        '''
        sqrt_actual = torch.sign(actual_reshaped[..., self.C + 3: self.C + 5]) * torch.sqrt(
                        torch.abs(actual_reshaped[..., self.C + 3: self.C + 5]) + 1e-6 )

        sqrt_preds  = torch.sign(prediction_reshaped[..., self.C + 3: self.C + 5]) * torch.sqrt(
                        torch.abs(prediction_reshaped[..., self.C + 3: self.C + 5]) + 1e-6)
        '''
        sqrt_actual = actual_reshaped[..., self.C + 3: self.C + 5]
        sqrt_preds  = prediction_reshaped[..., self.C + 3: self.C + 5]
        regression_loss += self.mse( torch.flatten(torch.multiply(actual_presence, sqrt_actual)), 
                                     torch.flatten(torch.multiply(actual_presence, sqrt_preds)) )
        regression_loss = self.lambda_obj * regression_loss

        #===========CONFIDENCE LOSS============
        # If an object is detected in the box, the confidence loss (measuring the objectness of the box)
        confidence_loss  = self.mse( torch.flatten(torch.multiply(actual_presence, actual_reshaped[..., self.C:self.C+1])), 
                                     torch.flatten(torch.multiply(actual_presence, prediction_reshaped[..., self.C:self.C+1])) )
        _noobj_conf      = self.mse( torch.flatten(torch.multiply((1 - actual_presence), actual_reshaped[..., self.C:self.C+1])), 
                                     torch.flatten(torch.multiply((1 - actual_presence), prediction_reshaped[..., self.C:self.C+1])) )
        
        confidence_loss += self.lambda_noobj * _noobj_conf

        #=========Classification Loss==========
        # Class of BOX
        classification_loss = self.mse( torch.flatten(torch.multiply(actual_presence, actual_reshaped[..., 0:self.C])),
                                        torch.flatten(torch.multiply(actual_presence, prediction_reshaped[..., 0:self.C])) )

        loss = regression_loss + confidence_loss + classification_loss
        return loss

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, C=20, B=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S[0], self.S[1], self.C + self.B * 5)
        target  = target.reshape(-1, self.S[0], self.S[1], self.C + self.B * 5)
        
        exists_box = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        # [c0..cn, p, x, y, w, h]
        box_predictions = exists_box * ( predictions[..., self.C+1: self.C+5] )

        box_targets = exists_box * target[..., self.C+1: self.C+5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = ( predictions[..., self.C: self.C+1] )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C: self.C+1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C: self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C: self.C+1], start_dim=1),
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
# %%
