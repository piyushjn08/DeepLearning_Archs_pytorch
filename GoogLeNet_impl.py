#%% Imports and declarations
import torch
import torch.nn as nn 
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from random import randrange

class conv_layer(nn.Module):
    def __init__(self, inchannels, out_channels, **kwargs):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(inchannels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, data):
        x = self.conv(data)
        x = self.bn(x)
        x = self.relu(x)

        return x

class inception_block(nn.Module):
    def __init__(self, inchannels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inception_block, self).__init__()
        self.output1 = conv_layer(inchannels, out_1x1, kernel_size=1, stride=1, padding=0)
        
        self.output2 = nn.Sequential(conv_layer(inchannels, red_3x3, kernel_size=1, stride=1, padding=0),
                                    conv_layer(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1))
        
        self.output3 = nn.Sequential(conv_layer(inchannels, red_5x5, kernel_size=1, stride=1, padding=0),
                                    conv_layer(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2))
        
        self.output4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    conv_layer(inchannels, out_1x1pool, kernel_size=1, stride=1, padding=0))
        
    def forward(self, data):
        return torch.cat([self.output1(data), self.output2(data), self.output3(data), self.output4(data)], 1)

class GoogleNet(nn.Module):
    def __init__(self, inchannels, outclasses):
        super(GoogleNet, self).__init__()
        self.conv1 = conv_layer(inchannels, 64, kernel_size=7, stride=2, padding=3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_layer(64, 192, kernel_size=3, stride=1, padding=1)
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1*1*1024, outclasses)
    
    def forward(self, batch):
        x = self.conv1(batch)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.maxp(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxp(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxp(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        output1 = self.linear(x)

        return output1

# Loadable Dataset, but not enough for GPU
class MyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __get_item__(self, index):
        return self.X[index], self.y[index]

# Currently working for Classification (because it calculates accuracy)
# To be implement for Regression

class pytorch_trainer: 
    def __init__(self, model, criteria, optimizer,
                    lr=0.1, lr_factor=0.1, lr_patience=5, device=None,
                    init_weights=False):
        '''
        device : "cuda" / "cpu"
        dynamic_lr, if set to true, learning rate will change with training
        '''
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer

        # Learning Rate
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    factor=self.lr_factor,
                                                                    patience=self.lr_patience,
                                                                    verbose=1)
        
        # Weights Initialization
        if init_weights:
            self.initialize_weights()
        
        # Trends
        self.training_loss = []
        self.validation_loss = []
    
        self.checkpointing = False
        self.checkpoint_path = ''

        self.device = device
        if(device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
    
    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
   
    def enable_checkpoiting(self, path, savebest=False):
        self.checkpointing = True
        self.checkpoint_path = path

    def save_model(self, epoc, value):
        checkpoint = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}
        save_string = f"{self.checkpoint_path}" + "/" + f"{epoc}-{value}.pth.tar"
        torch.save(checkpoint, save_string)
        
        
    def summary(self, input):
        macs, params = profile(self.model, inputs=(input, ))
        print("Macs:", round(macs/1000000,2), 'M', flush=True)
        print("Params:", round(params/1000000,2),'M', flush=True)

    class pytorch_dataset: # pre-process incoming data (used when some randomness is required in preprocessing)
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return self.X.shape[0]
        
        def __getitem__(self, index):
            return self.X[index], self.y[index]

    # Only applicable for classifier models (so far, because of accuracy calculation)
    def fit(self, X, y, epochs, batch_size=1,trainClass=None, 
             validation=[], validationClass=None, 
            verbose=1, shuffle=False, calculate_acc=True):

        self.batch_size = batch_size
        self.X = X
        self.y = y

        # Prepare Training Dataset
        if(trainClass is None):
            trainClass = self.pytorch_dataset(X,y)
        
        # Prepare Validation Dataset
        do_validation = False
        if(len(validation)==2):
            if(validationClass is None):
                validationClass = self.pytorch_dataset(validation[0], validation[1])
            do_validation = True

        self.train_dataLoader = DataLoader(trainClass, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        self.val_dataLoader = DataLoader(validationClass, batch_size=1, shuffle=False, num_workers=2)

        self.training_loss = []
        self.validation_loss = []

        bestTrainLoss = 9999999.0
        bestValLoss   = 9999999.0

        print("\nTraining on:", self.device, flush=True)
        
        # Run Training Loop
        for epoc in range(epochs):
            start_time = time.time()

            # Train
            total_loss, train_acc = self.__train_batch__(self.train_dataLoader, epoc, calculate_acc)
            self.training_loss.append(total_loss)

            # Validate
            if(do_validation):
                val_loss, val_acc = self.__validate_batch__(self.val_dataLoader, calculate_acc)
                self.validation_loss.append(val_loss)

            # Save Checkpoint

            time_taken = time.time() - start_time

            # Print Epoch Results
            # Use Flush = True to avoid messing with tqdm prints
            info_train = f"Epoch {epoc}: Time Taken:{round(time_taken,2)}s, loss:{round(self.training_loss[-1],2)}"
            info_val = f""

            if(calculate_acc):
                info_train = info_train + f", train_acc:{round(train_acc,2)}"
                if(do_validation):
                    info_val = info_val + f", val_acc:{round(val_acc,2)}"
            
            if(do_validation):
                info_val = f"val_Loss:{round(self.validation_loss[-1],2)}" + info_val
            else:
                info_val = f""
            
            print(info_train + info_val + "\n", flush=True)

    def __train_batch__(self, dataLoader, epoc, calculate_acc):
        self.model.train(True)
        batch_count = 0
        total_loss = 0
        correct_predictions = 0
        with tqdm(dataLoader, unit="batch", desc=("Epoch " + str(epoc))) as tepoch:
            for X, y in tepoch:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                batch_count = batch_count + 1

                # Forward Pass
                preds = self.model.forward(X)
                
                # Calculate Loss
                loss = self.criteria(preds, y) # Predictions, correct index in each prediction
                total_loss = total_loss + loss.item()

                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Batch Accuracy
                if calculate_acc:
                    y_categories = torch.argmax(preds, dim=1)
                    correct_predictions += (y_categories == y).long().sum().__float__()
                
                    tepoch.set_postfix({"loss":round(total_loss/batch_count,2), 
                                    "acc":round(correct_predictions/batch_count, 2)})
                else:
                    tepoch.set_postfix({"loss":round(total_loss/batch_count,2)})
        
        total_loss = total_loss/batch_count

        self.scheduler.step(total_loss)
        
        if calculate_acc:
            acc = correct_predictions/ float(batch_count)
            return total_loss, acc
        
        return total_loss, None
    
    def __validate_batch__(self, dataLoader, calculate_acc):
        self.model.Train(False)
        batch_count = 0
        total_loss = 0
        correct_predictions = 0
        for X, y in dataLoader:
            X = X.to(device=self.device)
            y = y.to(device=self.device)
            
            batch_count = batch_count + 1
            # Forward Pass
            preds = self.model.forward(X)
            loss = self.criteria(preds, y)
            total_loss = total_loss + loss.item()
            
            # Batch Accuracy
            if calculate_acc:
                y_categories = torch.argmax(preds, dim=1)
                correct_predictions += (y_categories == y).long().sum().float()
        
        validation_loss = total_loss / batch_count
        if calculate_acc:
            accuracy = correct_predictions.__float__() / float(dataLoader.__len__())
            return validation_loss, accuracy
        else:
            return validation_loss, None

#%% Test Model
classes = 10
set_count = 10
X = torch.randn(set_count, 3, 224, 224)
y = []
for i in range(set_count):
    index = randrange(classes)
    y.append(index)
y = torch.LongTensor(y)

print("Batch Shape:", X.shape, flush=True)

# Define Model
model = GoogleNet(3, classes)

# Define Tuning Parameters
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Initialize TrainerClass
trainer = pytorch_trainer(model, criteria, optimizer)
trainer.summary(X)

# Train Model
trainer.fit(X, y, epochs=50, batch_size=4, calculate_acc=True)
# %% train model

