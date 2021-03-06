import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import time
import os


matplotlib.use('Agg')


classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LOG_INTERVAL = 50
EPOCHS = 50

LR = 0.001
MM = 0.9
WD = 0.0001

kwargs = {'num_workers': 1, 'pin_memory': True}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=data_transforms['train'])

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=data_transforms['val'])

train_loader = DataLoader(
    trainset,
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs
)
test_loader = DataLoader(
    testset,
    batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DenseNet121
model_ft = models.densenet121(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, len(classes))
model = model_ft.to(device)

# ResNet50
model_ft = models.resnet50(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(classes))
model = model_ft.to(device)


# Adam
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train(model, device, train_loader, optimizer, epoch, exp_lr_scheduler):
    model = model.train()
    exp_lr_scheduler.step()
    loss_list = []
    acc_list = []

    train_loss_list = []
    train_acc_list = []

    for batch_idx, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        train_loss_list.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)

        acc = pred.eq(labels.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        train_acc_list.append(acc.item())

        if batch_idx % LOG_INTERVAL == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
            LOG_INFO(msg)
            loss_list.clear()
            acc_list.clear()
    return np.mean(np.array(train_loss_list)), np.mean(np.array(train_acc_list))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            test_loss += F.cross_entropy(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)


if __name__ == '__main__':
    test_loss, test_acc = [], []
    train_loss, train_acc = [], []
    for epoch in range(1, EPOCHS + 1):
        testloss, testacc = test(model, device, test_loader)
        test_loss.append(testloss)
        test_acc.append(testacc)

        trainloss, trainacc = train(
            model, device, train_loader, optimizer, epoch, exp_lr_scheduler)
        train_loss.append(trainloss)
        train_acc.append(trainacc)

    plt.figure()
    plt.plot(train_loss, label='train loss vs. iterations', color='green')
    plt.xlabel('iteration(s)')
    plt.ylabel("train loss")
    plt.title("train loss vs. iterations")
    plt.savefig('trainloss_den_adam1.png')
    plt.figure()
    plt.plot(train_acc, label='train accuracy vs. iterations', color='r')
    plt.xlabel('iteration(s)')
    plt.ylabel("train accuracy")
    plt.title("train accuracy vs. iterations")
    plt.savefig('trainacc_den_adam1.png')

    plt.figure()
    plt.plot(test_loss, label='test loss vs. epochs', color='green')
    plt.xlabel('epoch(s)')
    plt.ylabel("test loss")
    plt.title("test loss vs. epochs")
    plt.savefig('testloss_den_adam1.png')
    plt.figure()
    plt.plot(test_acc, label='test accuracy vs. epochs', color='r')
    plt.xlabel('epoch(s)')
    plt.ylabel("test accuracy")
    plt.title("test accuracy vs. epochs")
    plt.savefig('testacc_den_adam1.png')

    test(model, device, test_loader)
