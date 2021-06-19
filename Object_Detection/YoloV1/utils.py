#%% Imports and Declarations
import torch
import numpy as np
import cv2, time

class data_encoder_pytorch:
    def __init__(self, grid_dim):
        '''
        grid_dim = [Height, Width] or [Y,X] or [Rows, Columns]
        '''
        self.grid_dim = grid_dim

    def __cornered_to_centered__(self, batch):
        '''
        Batch Shape:  (batch_size, 4)
                      [x1, y1, x2, y2]
        Replace torch by numpy to make it independent from torch
        '''

        batch[..., 2] = batch[..., 2] - batch[..., 0] # Width
        batch[..., 3] = batch[..., 3] - batch[..., 1] # Height
        batch[..., 0] = batch[..., 0] + (batch[..., 2]/2.0) # X center
        batch[..., 1] = batch[..., 1] + (batch[..., 3]/2.0) # Y center

        return batch
    
    def __centered_to_cornered__(self, batch):
        '''
        Batch Shape:  (batch_size, 4)
                      [x, y, w, h]
        Replace torch by numpy to make it independent from torch
        '''

        batch[:, 2] = batch[:, 0] + batch[:, 2]
        batch[:, 3] = batch[:, 1] + batch[:, 3]

        return batch

    def annotaion_encoding(self, bbox, image_size, box_format='cornered'):
        '''
        Inputs:
            batch : (batch_size, 4) in torch.FloatTensor
                    [x1, y1, x2, y2]'cornered' or [x, y, w, h]'centered'
            image_size = (batch_size, 2) [height, width]
            box_format = 'cornered' / 'centered'
        
        output:
            batch : Xgrid, Ygrid, Xrel_c, Yrel_c, Xrel_w, Yrel_w
        '''
        centered_batch = bbox
        if box_format == 'cornered':
            centered_batch = self.__cornered_to_centered__(bbox)
        
        # Image and Grid Basic Parameters
        grid_dim  = self.grid_dim
        image_width  = image_size[:, 1]
        image_height = image_size[:, 0]
        gridx_count  = float(grid_dim[1])
        gridy_count  = float(grid_dim[0])

        Xcenter  = centered_batch[:, 0]
        Ycenter  = centered_batch[:, 1]
        Xwidth  = centered_batch[:, 2]
        Yheight = centered_batch[:, 3]

        # Grid Cell Height and Width
        grid_width  = image_width / gridx_count
        grid_height = image_height / gridy_count
        
        # Grid coordinates where center of annotated bounding box lies
        Xgrid = torch.floor(Xcenter / grid_width)
        Ygrid = torch.floor(Ycenter / grid_height)

        # Relative coordinated of center of Bounding box wrt to grid cell size
        Xrel_c = (Xcenter - (Xgrid*grid_width))  / grid_width
        Yrel_c = (Ycenter - (Ygrid*grid_height)) / grid_height

        # Relative width and height of bounding box wrt to grid cell size
        Xrel_w = Xwidth/grid_width
        Yrel_w = Yheight/grid_height

        return Xgrid, Ygrid, Xrel_c, Yrel_c, Xrel_w, Yrel_w
    
    def decode_grid(self, image_size, grid_coords, rel_center, rel_width):
        
        '''
        Input:
        Image Size     :  (batch_size, 2) [Y, X]
        grid_coord     :  (batch_size, 2) [X, Y]
        Relative Center:  (batch_size, 2) [X, Y]
        Relative Width :  (batch_size, 2) [X, Y]

        Output:
        bounding box coordinates [x1, y1, x2, y2] 'Cornered' format
        '''

        grid_dim = self.grid_dim
        
        image_width  = image_size[:, 1]
        image_height = image_size[:, 0]
        gridx_count  = float(grid_dim[1])
        gridy_count  = float(grid_dim[0])

        Xgrid        = grid_coords[:, 0]
        Ygrid        = grid_coords[:, 1]
        
        # Grid Cell Height and Width
        grid_width  = image_width / gridx_count
        grid_height = image_height / gridy_count

        # Grid leftmost point
        grid_X_coord = (grid_width  * Xgrid)
        grid_Y_coord = (grid_height * Ygrid)
        
        centerx_offset = (grid_width  * rel_center[:, 0])
        centery_offset = (grid_height * rel_center[:, 1])
        width          = (grid_width  * rel_width[:, 0])
        height         = (grid_height * rel_width[:, 1])

        X1 = torch.floor(grid_X_coord + centerx_offset - width/2).unsqueeze(1)
        Y1 = torch.floor(grid_Y_coord + centery_offset - height/2).unsqueeze(1)
        X2 = torch.floor(grid_X_coord + centerx_offset + width/2).unsqueeze(1)
        Y2 = torch.floor(grid_Y_coord + centery_offset + height/2).unsqueeze(1)

        stacked = torch.hstack((X1, Y1, X2, Y2))
        return stacked

    # Only supports one detection per image, to be implemented for multiple detections per image
    def to_grids(self, grid_coords, rel_center, rel_width, classes, class_count, grids=None):
        '''
        Call annotation_encoding to get all parameters
        Input:
        grid_coord     :  (batch_size, 2) [X, Y]
        Relative Center:  (batch_size, 2) [X, Y]
        Relative Width :  (batch_size, 2) [X, Y]
        classes        :  (batch_size, 1) [C]
        class_count    :  (batch_size, 1) [count]
        grid           :  (batch_size, S, S, C + B*5) # grid format already existing

        Output:
        bounding box coordinates [[xstart, ystart], [xend, yend]] 'Cornered' format
        '''
        batch_size = grid_coords.shape[0]
        if grids is None:
            grids = torch.zeros((batch_size, self.grid_dim[0], self.grid_dim[1], class_count + 5))

        for i in range(batch_size):
            gridx  = grid_coords[i, 0]
            gridy  = grid_coords[i, 1]
            cx     = rel_center[i, 0]
            cy     = rel_center[i, 1]
            dx     = rel_width[i, 0]
            dy     = rel_width[i, 1]

            target = classes[i]
            grids[i, gridy, gridx, target] = 1.0          # Class of object
            grids[i, gridy, gridx, class_count + 0] = 1.0 # Presence of Object 
            grids[i, gridy, gridx, class_count + 1] = cx  # Center X
            grids[i, gridy, gridx, class_count + 2] = cy  # Center Y
            grids[i, gridy, gridx, class_count + 3] = dx  # Width
            grids[i, gridy, gridx, class_count + 4] = dy  # Height
        
        return grids
    
    def show_image(self, dimentions, image, title='Image Display', wait=True,
                    color=(255,0,0), thickness=1):
        '''
        dimentions: [x1, y1, x2, y2]
        image     : OpenCV image
        '''
        xstart = dimentions[0]
        ystart = dimentions[1]
        xend   = dimentions[2]
        yend   = dimentions[3]

        cv2.rectangle(image, 
                        (xstart, ystart), 
                        (xend, yend), 
                        color, 
                        thickness)
        
        cv2.imshow(title, image)
        if(wait == True):
            key = cv2.waitKey(0)
            return key

    def resize_bb_coord(self, actual_im_size, target_im_size, bbox, format='cornered'):
        '''
        actual_im_size : (batch_size, 2) [Height, Width]
        target_im_size : [Height, Width]
        bbox           : (batch_size, 4), [x1, y1, x2, y2] or [x, y, w, h] :: cornered / centered
        '''
        
        if format == 'centered':
            bbox = self.__centered_to_cornered__(bbox)
        
        actual_im_size = torch.FloatTensor(actual_im_size)
        target_im_size = torch.FloatTensor([ [target_im_size[0], target_im_size[1]] for _ in range(actual_im_size.shape[0])])
        
        ratio_width  = (target_im_size[:, 1]/actual_im_size[:, 1]).detach().cpu().numpy()
        ratio_height = (target_im_size[:, 0]/actual_im_size[:, 0]).detach().cpu().numpy()

        bbox[:, 0] = bbox[:, 0] * ratio_width
        bbox[:, 1] = bbox[:, 1] * ratio_height
        bbox[:, 2] = bbox[:, 2] * ratio_width
        bbox[:, 3] = bbox[:, 3] * ratio_height

        return bbox.astype(np.int32)

class data_encoder:
    def __init__(self, grid_dim):
        '''
        grid_dim = [Height, Width] or [Y,X] or [Rows, Columns]
        '''
        self.grid_dim = grid_dim
    
    def __cornered_to_centered__(self, batch):
        '''
        Batch Shape:  (batch_size, 4)
                      [x1, y1, x2, y2]
        Replace torch by numpy to make it independent from torch
        '''

        batch[..., 2] = batch[..., 2] - batch[..., 0] # Width
        batch[..., 3] = batch[..., 3] - batch[..., 1] # Height
        batch[..., 0] = batch[..., 0] + (batch[..., 2]/2.0) # X center
        batch[..., 1] = batch[..., 1] + (batch[..., 3]/2.0) # Y center

        return batch
    
    def __centered_to_cornered__(self, batch):
        '''
        Batch Shape:  (batch_size, 4)
                      [x, y, w, h]
        Replace torch by numpy to make it independent from torch
        '''

        batch[:, 2] = batch[:, 0] + batch[:, 2]
        batch[:, 3] = batch[:, 1] + batch[:, 3]

        return batch

    def annotaion_encoding(self, batch, image_size, box_format='cornered'):
        '''
        Inputs:
            batch : (batch_size, 4) in torch.FloatTensor
                    [x1, y1, x2, y2]'cornered' or [x, y, w, h]'centered'
            image_size = [height, width]
            box_format = 'cornered' / 'centered'
        
        output:
            batch : [batch_size, 6]
                    [gridx, gridy, centerx, centery, rel_width, rel_height] 
        '''
        
        print(type(batch))
        centered_batch = batch
        if box_format == 'cornered':
            centered_batch = self.__cornered_to_centered__(batch)
        
        # Image and Grid Basic Parameters
        grid_dim  = self.grid_dim
        image_width  = float(image_size[1])
        image_height = float(image_size[0])
        gridx_count  = float(grid_dim[1])
        gridy_count  = float(grid_dim[0])

        Xcenter  = centered_batch[:, 0]
        Ycenter  = centered_batch[:, 1]
        Xwidth  = centered_batch[:, 2]
        Yheight = centered_batch[:, 3]

        # Grid Cell Height and Width
        grid_width  = image_width / gridx_count
        grid_height = image_height / gridy_count
        
        # Grid coordinates where center of annotated bounding box lies
        Xgrid = torch.floor(Xcenter / grid_width)
        Ygrid = torch.floor(Ycenter / grid_height)

        # Relative coordinated of center of Bounding box wrt to grid cell size
        Xrel_c = (Xcenter - (Xgrid*grid_width))  / grid_width
        Yrel_c = (Ycenter - (Ygrid*grid_height)) / grid_height

        # Relative width and height of bounding box wrt to grid cell size
        Xrel_w = Xwidth/grid_width
        Yrel_w = Yheight/grid_height

        return Xgrid, Ygrid, Xrel_c, Yrel_c, Xrel_w, Yrel_w

    def annotation_encoding_(self, annot_coord, image_size):
        '''
        INPUTS:
            annot_coord = [[X1, Y1], [X2, Y2]]
                        X1, Y1 = Top Left coordinate
                        X2, Y2 = Bottom Right coordinate

            image_size =  [height, width]
        OUTPUTS:
            grid_coord = [grid X, grid Y] # This shows the grid's location
            relative_center = [X center, Y center] # This shows the relative
                            center point of annotation box wrt width and height 
                            of each grid.
            relative_width  = [Width, Height] # This shows the relative width and height
                            of annotation box wrt width and height of each grid.

        '''
        # Image and Grid Basic Parameters
        grid_dim  = self.grid_dim
        image_width  = image_size[1]
        image_height = image_size[0]
        gridx_count  = grid_dim[1]
        gridy_count  = grid_dim[0]

        # Annotation Parameters basic and inferred
        Xstart  = annot_coord[0][0]
        Ystart  = annot_coord[0][1]
        Xend    = annot_coord[1][0]
        Yend    = annot_coord[1][1]
        Xwidth  = Xend - Xstart
        Yheight = Yend - Ystart
        Xcenter = (Xstart + Xend) / 2
        Ycenter = (Ystart + Yend) / 2

        # Grid Cell Height and Width
        grid_width  = image_width / gridx_count
        grid_height = image_height / gridy_count
        
        # Grid coordinates where center of annotated bounding box lies
        Xgrid = np.floor(Xcenter / grid_width)
        Ygrid = np.floor(Ycenter / grid_height)
        grid_coord = [int(Xgrid), int(Ygrid)]

        # Relative coordinated of center of Bounding box wrt to grid cell size
        Xrel_c = (Xcenter - (Xgrid*grid_width))  / grid_width
        Yrel_c = (Ycenter - (Ygrid*grid_height)) / grid_height
        relative_center = [Xrel_c, Yrel_c]

        # Relative width and height of bounding box wrt to grid cell size
        Xrel_w = (Xend - Xstart)/grid_width
        Yrel_w = (Yend - Ystart)/grid_height
        relative_width = [Xrel_w, Yrel_w]

        return grid_coord, relative_center, relative_width

    def decode_grid(self, image_size, grid_coords, rel_center, rel_width):
        
        '''
        Input:
        Image Size     :  [Y, X]
        grid_coord     :  (batch_size, 2) [X, Y]
        Relative Center:  (batch_size, 2) [X, Y]
        Relative Width :  (batch_size, 2) [X, Y]

        Output:
        bounding box coordinates [[xstart, ystart], [xend, yend]] 'Cornered' format
        '''

        grid_dim = self.grid_dim
        
        image_width  = float(image_size[1])
        image_height = float(image_size[0])
        gridx_count  = float(grid_dim[1])
        gridy_count  = float(grid_dim[0])

        Xgrid        = grid_coords[:, 0]
        Ygrid        = grid_coords[:, 1]
        
        # Grid Cell Height and Width
        grid_width  = image_width / gridx_count
        grid_height = image_height / gridy_count

        # Grid leftmost point
        grid_X_coord = (grid_width  * Xgrid)
        grid_Y_coord = (grid_height * Ygrid)
        
        centerx_offset = (grid_width  * rel_center[:, 0])
        centery_offset = (grid_height * rel_center[:, 1])
        width          = (grid_width  * rel_width[:, 0])
        height         = (grid_height * rel_width[:, 1])

        X1 = torch.floor(grid_X_coord + centerx_offset - width/2).unsqueeze(1)
        Y1 = torch.floor(grid_Y_coord + centery_offset - height/2).unsqueeze(1)
        X2 = torch.floor(grid_X_coord + centerx_offset + width/2).unsqueeze(1)
        Y2 = torch.floor(grid_Y_coord + centery_offset + height/2).unsqueeze(1)

        stacked = torch.hstack((X1, Y1, X2, Y2))
        return stacked

    def decode_grid_(self, image_size, grid_coord, rel_center, rel_width):
        '''
        Input:
        Image Size     :  [Y, X]
        grid_coord     :  [Y, X]
        Relative Center:  [X, Y]
        Relative Width :  [X, Y]

        Output:
        bounding box coordinates [[xstart, ystart], [xend, yend]]
        '''
        
        grid_dim = self.grid_dim
        
        image_width  = image_size[1]
        image_height = image_size[0]
        gridx_count  = grid_dim[1]
        gridy_count  = grid_dim[0]
        Xgrid        = grid_coord[1]
        Ygrid        = grid_coord[0]

        # Grid Cell Height and Width
        grid_width  = image_width / gridx_count
        grid_height = image_height / gridy_count

        # Grid leftmost point
        grid_X_coord = (grid_width  * Xgrid)
        grid_Y_coord = (grid_height * Ygrid)

        centerx_offset = (grid_width  * rel_center[0])
        centery_offset = (grid_height * rel_center[1])
        width          = (grid_width  * rel_width[0])
        height         = (grid_height * rel_width[1])

        X1 = int(grid_X_coord + centerx_offset - width/2)
        Y1 = int(grid_Y_coord + centery_offset - height/2)
        X2 = int(grid_X_coord + centerx_offset + width/2)
        Y2 = int(grid_Y_coord + centery_offset + height/2)

        return [X1,Y1], [X2,Y2]

    def resize_bb_coord(self, actual_im_size, target_im_size, bbox, format='cornered'):
        '''
        actual_im_size : [Height, Width]
        target_im_size : [Height, Width]
        bbox           : [x1, y1, x2, y2] or [x, y, w, h] :: cornered / centered
        '''
        
        if format == 'centered':
            bbox = self.__centered_to_cornered__(bbox)
        
        ratio_width  = float(target_im_size[1])/float(actual_im_size[1])
        ratio_height = float(target_im_size[0])/float(actual_im_size[0])

        bbox[:, 0] = bbox[:, 0] * ratio_width
        bbox[:, 1] = bbox[:, 1] * ratio_height
        bbox[:, 2] = bbox[:, 2] * ratio_width
        bbox[:, 3] = bbox[:, 3] * ratio_height

        return bbox

    def resize_bb_coord_(self, actual_im_size, target_im_size, bbox):
        '''
        actual_im_size : [Height, Width]
        target_im_size : [Height, Width]
        bbox           : [[Top Left], [bottom Right]]
        '''

        xstart = bbox[0][0]
        ystart = bbox[0][1]
        xend   = bbox[1][0]
        yend   = bbox[1][1]

        ratio_width  = target_im_size[1]/actual_im_size[1]
        ratio_height = target_im_size[0]/actual_im_size[0]

        xs = int(xstart * ratio_width)
        ys = int(ystart * ratio_height)
        xe = int(xend   * ratio_width)
        ye = int(yend   * ratio_height)

        dimentions = [[xs, ys], [xe, ye]]
        return dimentions

    def bb_presence_grid_coordinates(self, prediction, threshold):
        '''
        target: Single matrix output from model [Y, X, Channels]
        threshold: Minimum confidence value (in double)
        '''
        bb_details = []
        height, width, channels = prediction.shape
        for y in range(height):
            for x in range(width):
                if(prediction[y][x][0] >= threshold): # Channel 0 marks bb presence
                    grid_coordinate = [y, x]
                    relative_center = [prediction[y][x][1], prediction[y][x][2]] # [X, Y]
                    relative_dimention = [prediction[y][x][3], prediction[y][x][4]] # [X, Y]
                    bb_details.append([grid_coordinate, relative_center, relative_dimention])
        
        return np.array(bb_details)

    def show_image(self, dimentions, image, title='Image Display', wait=True,
                    color=(255,0,0), thickness=1):
        '''
        dimentions: [[xstart, ystart], [xend, yend]]
        image     : OpenCV image
        '''
        xstart = dimentions[0][0]
        ystart = dimentions[0][1]
        xend   = dimentions[1][0]
        yend   = dimentions[1][1]

        cv2.rectangle(image, 
                        (xstart, ystart), 
                        (xend, yend), 
                        color, 
                        thickness)
        
        cv2.imshow(title, image)
        
        if(wait == True):
            key = cv2.waitKey(0)
            return key

if __name__ == '__main__':
    encoder = data_encoder_pytorch((10,10))
    image_size = torch.FloatTensor([[100,100], 
                                    [100, 100]])
    coordinates = [[0.0, 0.0, 9.0, 39.0], 
                   [30.0, 30.0, 60.0, 60.0]]
    coordinates = torch.FloatTensor(coordinates)

    start_time = time.time()
    Xgrid, Ygrid, Xrel_c, Yrel_c, Xrel_w, Yrel_w = encoder.annotaion_encoding(coordinates, image_size)
    grid_coord = torch.hstack((Xgrid.unsqueeze(1), Ygrid.unsqueeze(1)))
    rel_center = torch.hstack((Xrel_c.unsqueeze(1), Yrel_c.unsqueeze(1)))
    rel_width  = torch.hstack((Xrel_w.unsqueeze(1), Yrel_w.unsqueeze(1)))

    coords = encoder.decode_grid(image_size, grid_coord, rel_center, rel_width)
    print(coords)
    print("Time Taken:", time.time() - start_time)
    
# %%
