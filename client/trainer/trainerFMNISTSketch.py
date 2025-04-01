from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#TESTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from .sketch_utils import delta_weights, compress, get_random_hashfunc, get_params, decompress, set_params_fedsketch
from .malicious_utils import FMNIST_label_flipping

class TrainerFMNISTSketch:
    def __init__(self,num_id, mode) -> None:
        # id and model
        self.num_id = num_id
        self.mode = mode
        self.model = self.define_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # split data
        # select a random number ranging from 10000 < num_samples < 20000
        self.num_samples = int(np.random.choice(np.arange(10000, 20000, 1000)))
        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = self.split_data()
        self.stop_flag = False
        self.args = None

        self.weights = get_params(self.model)
        self.old_weights = get_params(self.model)
        self.compression = 0.0066666666
        self.learning_rate = 1e-3
        self.global_learning_rate = 1
        self.length = 20
        self.vector_length = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.index_hash_function = [get_random_hashfunc(_max=int(self.compression*self.vector_length), seed=repr(j).encode()) for j in range(self.length)]
        self.metrics_names = ["accuracy"]

    def set_args(self,args):
        self.args = args

    def get_num_samples(self):
        return self.num_samples

    def define_model(self, input_shape=(28, 28, 1), n_classes=10):
        model = LeNet5(num_classes=10)

        return model

    def split_data(self):
        batch_size = 64

        train_dataset = datasets.FashionMNIST(root="dataset/", download=True, train=True, transform=transforms.Compose([
                                                                                            transforms.Resize((32,32)),
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize(mean=(0.5,), std=(0.5,))]))
        
        #Trainers are numbered from 0 to N. Choose how many trainer you want to be malicious
        if(self.num_id == 0):
            FMNIST_label_flipping(train_dataset)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = datasets.FashionMNIST(root="dataset/", download=True, train=False, transform=transforms.Compose([
                                                                                            transforms.Resize((32,32)),
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize(mean=(0.5,), std=(0.5,))]))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataset, test_dataset, train_loader, test_loader

    def train_model(self):
        self.old_weights = get_params(self.model)
        
        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()

            for inputs, labels in self.train_loader:

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()


            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

        self.weights = get_params(self.model)

    def eval_model(self):
        self.model.eval()
        correct = 0
        total = 0
        acc = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                total += predicted.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct/total
        
        return acc
    
    def all_metrics(self):
        metrics_names = self.metrics_names
        values = [self.eval_model()]
        return dict(zip(metrics_names, values))

    def get_weights(self):
        delta = delta_weights(self.weights, self.old_weights)
        self.sketch = compress(delta, self.compression, self.length, 1, 90, self.index_hash_function)
        return self.sketch

    def update_weights(self, global_sketch):
        n_weights = decompress(self.weights,global_sketch, len(global_sketch),-10000, 10000,self.index_hash_function)
        for k,v in n_weights.items():
 
            n_weights[k] = v*self.global_learning_rate + self.old_weights[k]

        
        set_params_fedsketch(self.model, n_weights)
        self.model = self.model.float()

    def set_stop_true(self):
        self.stop_flag = True

    def get_stop_flag(self):
        return self.stop_flag

class LeNet5(nn.Module):
    def __init__(self, num_classes,channels=1):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, 6*channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6*channels, 16*channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400*channels, 120*channels)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120*channels, 84*channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84*channels, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out