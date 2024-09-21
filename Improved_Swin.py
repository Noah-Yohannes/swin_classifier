###         Swin Transformer based for Image Classification
#Importing Libraris

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn import functional as F
from torcheval.metrics.functional import multiclass_precision, multiclass_accuracy, multiclass_f1_score
from torchvision.transforms import v2

import numpy as np
import pandas as pd
import os
from skimage import io #
from sklearn.metrics import f1_score,precision_score, accuracy_score

# MEAN and STD to normalise images
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

model = torchvision.models.swin_b(weights="DEFAULT",progress=True)
images_transform = v2.Compose([
    v2.Resize(224),
    v2.Normalize(mean=MEAN, std=STD),
    v2.ToImageTensor(),
    v2.ConvertImageDtype(dtype=torch.float32),
])

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

combined_path = "Combined-Garbage-Dataset"
train_csv = "garbage_training_images.csv"
test_csv = "garbage_testing_images.csv"

train_dataset = SwinDataSet(combined_path,train_csv, images_transform)
test_dataset = SwinDataSet(combined_path, test_csv, images_transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

class SwinModel(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        super(SwinModel, self).__init__()
        self.swin = model
        self.num_features = self.swin.head.in_features
        # Replace the classification head
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
        return self.swin(x)
    def predict(self, x):
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes

num_classes = 10
SwinClassifier = SwinModel(num_classes = 10,freeze_base=False)
#SwinClassifier = SwinClassifier.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SwinClassifier = SwinClassifier.to(device)

epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(SwinClassifier.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)

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
        # added begin
        logits = SwinClassifier(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss #.item()
        preds = torch.argmax(logits, dim=1)
        predicted.extend(preds.cpu().numpy())
        actual.extend(y_batch.cpu().numpy())
        # added end
        #logits = SwinClassifier(x_batch)
        #loss = criterion(logits, y_batch)
        #epoch_loss +=loss.item()
        # backward-pass
        #loss.backward()
        # update weights
        #output = SwinClassifier.predict(x_batch)
        #optimizer.step()
        print("Output : ", preds)
        print("Loss : ", loss)
        print("y_batch : ", y_batch)
        #actual.append(y_batch.cpu().detach().numpy())
        #predicted.append(output.cpu().detach().numpy())
        #train_acc = torch.sum(output == y_batch)

    epoch_loss /= len(t_data_loader)
    predicted = torch.tensor(predicted)
    actual = torch.tensor(actual)
    epoch_accuracy = multiclass_accuracy(predicted, actual, num_classes=10)
    epoch_precision = multiclass_precision(predicted, actual, num_classes=10)
    epoch_f1 = multiclass_f1_score(predicted, actual, num_classes=10)
    return epoch_loss, epoch_accuracy.item(), epoch_precision.item(), epoch_f1.item()

    #actual = np.concatenate(actual, axis=0)
    #predicted = np.concatenate(predicted, axis=0)
    #epoch_accuracy = multiclass_accuracy(actual, predicted, num_classes=10) #accuracy_score(actual,predicted)
    #epoch_loss = epoch_loss / len(t_data_loader)
    #epoch_precision = multiclass_precision(actual, predicted, num_classes=10) #precision_score(actual,predicted, average='weighted')
    #epoch_f1 = multiclass_f1_score(actual, predicted, num_classes=10) #f1_score(actual,predicted, average='weighted')
    #return epoch_loss, epoch_accuracy, epoch_precision, epoch_f1

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
            loss = criterion(logit, y_batch).item()
            #output = SwinClassifier.predict(x_batch)
            #test_acc = torch.sum(output == y_batch)
            epoch_loss += loss
            preds = torch.argmax(logit, dim=1)
            predicted.extend(preds.cpu().numpy())
            actual.extend(y_batch.cpu().numpy())
            print("Validation loss : ",loss)
            print("Validation y_batch :  ", y_batch)
            print("Validation output : ",preds)
            #actual.append(y_batch.cpu().detach().numpy())
            #predicted.append(output.cpu().detach().numpy())
            #actual = np.concatenate(actual, axis=0)
            #predicted = np.concatenate(predicted, axis=0)
    epoch_loss /= len(v_data_loader)
    predicted = torch.tensor(predicted)
    actual = torch.tensor(actual)
    epoch_accuracy = multiclass_accuracy(predicted, actual, num_classes=10)
    epoch_precision = multiclass_precision(predicted, actual, num_classes=10)
    epoch_f1 = multiclass_f1_score(predicted, actual, num_classes=10)

    #epoch_loss = epoch_loss / len(v_data_loader)
    #epoch_accuracy = accuracy_score(actual,predicted)
    #epoch_precision = precision_score(actual,predicted,average="weighted")
    #epoch_f1 = f1_score(actual,predicted, average="weighted")
    return epoch_loss, epoch_accuracy.item(), epoch_precision.item(), epoch_f1.item()
with open("improved_swin_result.txt", 'w') as f:
    for i in range (epochs):
         print(f"Epoch {i} : \t Loss \t Accuracy  \t Precision  \t F1-score ")
         tr_loss, tr_acc, tr_prec, tr_f1score = training_func(train_dataloader)
         print(str(tr_loss) + '  \t  '+ str(tr_acc) +'  \t  ' + str(tr_prec) + '  \t  ' + str(tr_f1score))
         f.writelines(str(tr_loss) + '  \t  '+ str(tr_acc) +'  \t  ' + str(tr_prec) + '  \t  ' + str(tr_f1score))
         test_loss, test_acc, test_prec, test_f1score = validation_func(test_dataloader)
         print(str(test_loss) + '  \t  '+ str(test_acc) + '  \t  '+ str(test_prec) + '  \t  '+ str(test_f1score))
         f.writelines(str(test_loss) + '  \t  '+ str(test_acc) + '  \t  '+ str(test_prec) + '  \t  '+ str(test_f1score))
