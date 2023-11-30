import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

#Create models below
class CNN0(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()      
        self.CNN0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(4096, 32),
            nn.LeakyReLU(),
            nn.Linear(32, class_cnt)
        )
        
    def forward(self, x):
        logits = self.CNN0(x)
        return logits
    
class CNN1(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()      
        self.CNN1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.Dropout(.05),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.Dropout(.05),
            nn.Sigmoid(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(4096, 32),
            nn.Dropout(.05),
            nn.Sigmoid(),
            nn.Linear(32, class_cnt)
        )
        
    def forward(self, x):
        logits = self.CNN1(x)
        return logits
    
class CNN2(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()      
        self.CNN2 = nn.Sequential(
            nn.Linear(32, 3, True),
            nn.ReLU(),
            nn.Linear(3, 64, True),
            nn.ReLU(),
            nn.Linear(64, 64, True),
            nn.ReLU(),
            
            nn.Linear(64, 64, True),
            nn.ReLU(),
            nn.Linear(64, 64, True),
            nn.ReLU(),
            nn.Linear(64, 64, True),
            
            nn.ReLU(),
            
            nn.Linear(64, 64, True),
            nn.ReLU(),
            nn.Linear(64, 64, True),
            nn.ReLU(),
            nn.Linear(64, 64, True),
            
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(6144, class_cnt)
        )
        
    def forward(self, x):
        logits = self.CNN2(x)
        return logits
    
class CNN3(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()      
        self.CNN3 = nn.Sequential(
            nn.Conv2d(3, 64, 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=2),
            nn.ReLU(),
            
            nn.MaxPool2d(5),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )
        
    def forward(self, x):
        logits = self.CNN3(x)
        return logits

#Returns all model names, we can pick from this list later
def get_approach_names():
    approach_names = ["CNN0", "CNN1", "CNN2", "CNN3"]
    return approach_names

#Returns the description of all layers, used in our evaluation
def get_approach_description(approach_name):
    desc = {
        "CNN0" : "14 Layer CNN with Leaky ReLU as the activation function. Uses convolution layers to transform the data.",
        "CNN1" : "17 Layer CNN with Sigmoid as the activation function. Uses convolution layers to transform the data. Also has a dropout rate of 5%. Data is randomly flipped horizontally.",
        "CNN2" : "20 Layer CNN with ReLu as the activation function. Uses linear layers to transform the data. Data is randomly flipped vertically.",
        "CNN3" : "17 Layer CNN with ReLu as the activation function. Uses convolution to transform the data. One of the convolution layers has a stride length of 2."
    }
    
    return desc[approach_name]

#Data augmenation
def get_data_transform(approach_name, training):
    if (not training or approach_name == "CNN0" or approach_name == "CNN3"):
        data_transform = v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertDtype()
        ])
    #Data is randomly flipped horizontally
    elif (approach_name == "CNN1"):
        data_transform = v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertDtype(),
        v2.RandomHorizontalFlip()
        ])   
    #Data is randomly flipped vertically
    elif (approach_name == "CNN2"):
        data_transform = v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertDtype(),
        v2.RandomVerticalFlip()
        ])
    
    return data_transform

#Returns batch size based on model
def get_batch_size(approach_name):
    if (approach_name == "CNN0"):
        batch_size = 64
    elif (approach_name == "CNN1"):
        batch_size = 128
    elif (approach_name == "CNN2"):
        batch_size = 256
    else:
        batch_size = 32
    
    return batch_size

#Creates and returns model
def create_model (approach_name, class_cnt):
    if (approach_name == "CNN0"):
        model = CNN0(class_cnt)
    elif (approach_name == "CNN1"):
        model = CNN1(class_cnt)
    elif (approach_name == "CNN2"):
        model = CNN2(class_cnt)
    elif (approach_name == "CNN3"):
        model = CNN3(class_cnt)
    return model

#Trains the model
def train_model (approach_name, model, device, train_dataloader, test_dataloader):
    size = len(train_dataloader.dataset)
    model.train()
    
    epoch = 10
    
    for i in range (epoch):
        print(f"===== EPOCH {i + 1} =====")
        for _, (X,y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
    