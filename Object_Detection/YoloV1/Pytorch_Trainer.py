import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from torchprofile import profile_macs
from torchstat import stat
'''
Functionalities
1. Auto Assign DataLoader
2. Print Summary of Model
3. Trace Model for output shapes
4. Wrapped Training and Testing codes
5. Easy CPU, GPU shifting
'''

'''
To Do : 
+ Criteria may have multiple parameters as input, find a user defined workaourd for loss calcuation requiring more than 2 inupts
+ Loss calculation after complete batch has run (right now NN updates every batch)
'''

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
        
        self.model = self.model.to(self.device)
        self.addl_fn = self.addl_criteria()

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
    
    def clamp_grads(self, clip_value):
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def enable_checkpointing(self, path, save_best=False):
        self.checkpointing = True
        self.checkpoint_path = path
        self.save_best = save_best

    def save_model(self, epoc, value):
        checkpoint = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}
        save_string = f"{self.checkpoint_path}" + "/" + f"{epoc}-{value}.pth.tar"
        torch.save(checkpoint, save_string)
        
        
    def summary(self, input_shape):
        shape = tuple([1] + list(input_shape))
        sample_input = torch.randn(shape).to(device='cpu')
        self.model = self.model.to(device='cpu')
        #macs, params = profile(self.model, inputs=(sample_input,))
        macs = profile_macs(self.model, sample_input)
        stat(self.model, input_shape)
        print("Macs:", round(macs/1000000,2), 'M', flush=True)
        
        self.model = self.model.to(device=self.device)
        #print("Params:", round(params/1000000,2),'M', flush=True)

    class pytorch_dataset: # pre-process incoming data (used when some augmentation is required in preprocessing)
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return self.X.shape[0]
        
        def __getitem__(self, index):
            return self.X[index], self.y[index]

    class addl_criteria:
        def __init__(self):
            pass
        def start(self):
            pass
        def between(self, preds, actual):
            pass
        def end(self):
            pass
        

    def fit(self, X, y, epochs, batch_size=1,trainClass=None, 
             validation=[], validationClass=None,
             verbose=1, shuffle=False, calculate_acc=False, 
             anomaly_detection=False, addl_fn=None):
        
        '''
        trainClass: Class defining any preprocessing required before sending data from training in batches
        validationClass: Same as trainClass but for validation
        checkpoint_path: Path where checkpoint has to be saved
        save_best:  save only if loss is improved
        calculate_acc: Calculate accuracy of categorical output
        anomaly_detection: Debug mode for NaN outputs
        addl_fn: class defining function as start, between and end of epoch
        '''

        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.anomaly_detection = anomaly_detection
        
        # Set Training Configurations
        print("\nTraining on:", self.device, flush=True)
        if(self.anomaly_detection):
            print("Anomaly Detection Mode: ON")
            torch.autograd.set_detect_anomaly(True)
        else:
            torch.autograd.set_detect_anomaly(False)
        
        previous_weights = []
        for param in self.model.parameters():
            previous_weights.append(param.view(-1))
        previous_weights = torch.cat(previous_weights)
        
        if(addl_fn is not None):
            self.addl_fn = addl_fn
        
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

        best_train_loss = None
        best_val_loss   = None
        best_loss   = None


        # Run Training Loop
        for epoc in range(epochs):
            start_time = time.time()

            ################## Train ##################
            total_loss, train_acc = self.__train_batch__(self.train_dataLoader, epoc, calculate_acc)
            self.training_loss.append(total_loss)
            
            ################### Audit Weight Changes ######################
            current_weights = []
            for param in self.model.parameters():
                current_weights.append(param.view(-1))
            current_weights = torch.cat(current_weights)
            
            if(torch.all(current_weights.eq(previous_weights))):
                print("There is no change in weights, Model might have reached to its local/global minimum")
            previous_weights = current_weights
            
            ################## Validation #########################
            if(do_validation):
                val_loss, val_acc = self.__validate_batch__(self.val_dataLoader, calculate_acc)
                self.validation_loss.append(val_loss)

            ################## Save checkpoint #####################
            if self.checkpointing == True: # saving enabled
                if(self.save_best):
                    if best_loss is None:
                        print(f"saving model {epoc}-{round(total_loss, 2)}.pth.tar", flush=True)
                        self.save_model(epoc, round(total_loss, 2))
                        best_loss = total_loss
                    elif best_loss > total_loss:
                        print(f"Model improved from {best_loss} to {total_loss} saving model {epoc}-{round(total_loss, 2)}.pth.tar", flush=True)
                        self.save_model(epoc, round(total_loss, 2))
                        best_loss = total_loss
                else:
                    print(f"saving model {epoc}-{round(total_loss, 2)}.pth.tar", flush=True)
                    self.save_model(epoc, round(total_loss, 2))
            
            time_taken = time.time() - start_time

            ################## Print Epoch Results #######################
            # Use Flush = True to avoid messing with tqdm prints
            info_train = f"\nEpoch {epoc}: Time Taken:{round(time_taken,2)}s, loss:{round(self.training_loss[-1], 4)}"
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
        self.model = self.model.train(True)
        batch_count = 0
        total_loss = 0
        correct_predictions = 0
        self.addl_fn.start()
        with tqdm(dataLoader, unit="batch", desc=("Epoch " + str(epoc))) as tepoch:
            for X, y in tepoch:
                X = X.to(device=self.device)
                y = y.to(device=self.device)

                batch_count = batch_count + 1
                # Forward Pass
                preds = self.model.forward(X)
                # Calculate Loss
                loss = self.criteria.forward(preds, y) # Predictions, correct index in each prediction
                total_loss = total_loss + loss.item()
                self.addl_fn.between(preds, y)

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
        self.addl_fn.end()

        if calculate_acc:
            acc = correct_predictions/ float(batch_count)
            return total_loss, acc
        
        return total_loss, None
    
    def __validate_batch__(self, dataLoader, calculate_acc):
        self.model = self.model.eval()
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