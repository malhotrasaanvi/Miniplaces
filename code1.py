# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.conv1 = nn.Conv2d(3, out_channels =6, kernel_size = 5, stride=1)
        self.b = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv2 = nn.Conv2d(6, out_channels = 16, kernel_size = 5, stride=1)
        self.c = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.fl = nn.Flatten()

        self.lin1 = nn.Linear(in_features= 400,out_features=256, bias=True)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features =256,out_features=128, bias=True)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(128, 100, bias=True)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.b(out1)
        out1 = self.max_pool1(out1)

        out2 = self.conv2(out1)
        out2 = self.c(out2)
        out2 = self.max_pool2(out2)

        out3 = self.fl(out2)
        out3 = self.lin1(out3)
        
        out3 = self.relu1(out3)

        out4 = self.lin2(out3)
        out4 = self.relu2(out4)

        out5 = self.lin3(out4)

        shape_dict = {
            1: list(out1.shape),
            2: list(out2.shape),
            3: list(out3.shape),
            4: list(out4.shape),
            5: list(out5.shape)
        }
        outputs = [out1, out2, out3, out4, out5]

        return out5, shape_dict
        

    

    


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            train = 1
            for dimension in param.shape:
                train += dimension
            ##model_params += param.numel()
            model_params += train
    model_params /= 1e6
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
