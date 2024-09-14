
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn import functional as F

from torchvision.transforms import v2

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from skimage import io #
from sklearn.metrics import f1_score, precision_score, accuracy_score

# MEAN and STD to normalise images
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

### Images Transform
images_transform = v2.Compose([
    v2.Resize(224),
    v2.Normalize(mean=MEAN, std=STD),
    v2.ToImageTensor(),
    v2.ConvertImageDtype(dtype=torch.float32),
])

### Datasets and DataLoaders
class SwinDataSet(Dataset):
    def __init__(self, file_path, csv_file, transform=None) :
        self.file_path = file_path
        self.transform = transform 
        self.csv_file = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self,index) :
        #navigate to the folder & obtain pic name from the csv file (df) index
        image_path = os.path.join(self.file_path, self.csv_file.iloc[index,0])  
        # image to be returned
        image_instance = io.imread(image_path)
        # corresponding label to be returned
        class_label = self.csv_file.iloc[index, 1] 
        if self.transform:
            image_instance = self.transform(image_instance)

        return (image_instance,class_label)
### Loading Images
#- The images have already been preprocessed and separated in to training and testing files 

combined_path = "C:/Users/hoany/Desktop/Career_Opportunities/Job position Dr. Shabir/Project related/Combined-Custom-Dataset"
train_csv = "C:/Users/hoany/Desktop/Career_Opportunities/Job position Dr. Shabir/Project related/training_images.csv"
test_csv = "C:/Users/hoany/Desktop/Career_Opportunities/Job position Dr. Shabir/Project related/testing_images.csv"

train_dataset = SwinDataSet(combined_path,train_csv, images_transform)
test_dataset = SwinDataSet(combined_path, test_csv, images_transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

### Classification Model
class SwinModel(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        super(SwinModel, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT", progress=True) 
        self.mobilenet_features = self.mobilenet.features
        self.channel_adjust = nn.Conv2d(in_channels=576, out_channels=3, kernel_size=1)
        self.swin = torchvision.models.swin_b(weights="Swin_B_Weights.DEFAULT" ,progress=True) # using trained Swin_B weights so that the model uses trained weights
        self.num_features = self.swin.head.in_features
        # Replacing the classification head
        self.swin.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        if freeze_base:
            for param in self.swin.parameters():
                param.requires_grad = False
            for param in self.swin.head.parameters():
                param.requires_grad = True
        
    def forward(self,x):
        feature_map = self.mobilenet_features(x)
        # Adjust the channel size from 576 to 3 using 1x1 conv
        adjusted_features = self.channel_adjust(feature_map)
        output = self.swin(adjusted_features)
        return output 
    def predict(self, x):
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes

num_classes = 3
SwinClassifier = SwinModel(num_classes ,freeze_base=False) 

### Selecting device to run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SwinClassifier.to(device)

### Training hyperparameters
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(SwinClassifier.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

### Defining Training and Validation functions
def training_func(t_data_loader):    
    SwinClassifier.train()
    avg_train_loss = 0 
    epoch_loss = 0
    actual = []
    predicted = []
    
    for batch in t_data_loader:
        x_batch, y_batch = batch[0].to(device),batch[1].to(device)
        # set parameter gradients to zero
        optimizer.zero_grad()
        logits = SwinClassifier(x_batch)
        loss = criterion(logits, y_batch)
        epoch_loss +=loss
        # backward-pass
        loss.backward()
        # update weights
        output = SwinClassifier.predict(x_batch)
        optimizer.step()
        # Adding actual and predicted value to list
        actual.append(y_batch)
        predicted.append(output)
        # print("Output: ", output)
        # print("Truth: ", y_batch)
    epoch_accuracy = accuracy_score(actual,predicted)
    epoch_loss = epoch_loss / len(train_dataset)
    epoch_precision = precision_score(actual,predicted)
    epoch_f1 = f1_score(actual,predicted)
    
    return epoch_loss, epoch_accuracy, epoch_precision, epoch_f1  
        
def validation_func(v_data_loader):   
    SwinClassifier.eval()
    avg_test_loss = 0 
    epoch_loss =0
    actual = []
    predicted = []
    
    with torch.no_grad():
        for batch in v_data_loader:
            x_batch, y_batch = batch[0].to(device),batch[1].to(device)
            logit = SwinClassifier(x_batch)
            loss = criterion(logit, y_batch)
            output = SwinClassifier.predict(x_batch)
            epoch_loss += loss
            # Adding actual and predicted value to list
            actual.append(y_batch)
            predicted.append(output)
    epoch_loss = epoch_loss / len(test_dataset)
    epoch_accuracy = accuracy_score(actual,predicted)
    epoch_precision = precision_score(actual,predicted)
    epoch_f1 = f1_score(actual,predicted)
    
    return epoch_loss, epoch_accuracy, epoch_precision, epoch_f1  
    
for i in range (epochs):
    print(f"Epoch {i} : \t Loss \t Accuracy  \t Precision  \t F1-score ")
    tr_loss, tr_acc, tr_prec, tr_f1score = training_func(train_dataloader)
    print(tr_loss+ '  \t  '+ tr_acc +'  \t  '+ tr_prec+ '  \t  '+tr_f1score)
    test_loss, test_acc, test_prec, test_f1score = validation_func(test_dataloader)
    print(test_loss+ '  \t  '+ test_acc +'  \t  '+ test_prec+ '  \t  '+test_f1score)