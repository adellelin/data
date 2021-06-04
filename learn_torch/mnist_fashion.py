import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import namedtuple
import json
import os

from Inception_v3_full import Inception3
from data import FootballDataset
from models import Alexnet, Inceptionnet, Resnet, NeuralNetwork


_InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])

image_size = 224
model_type = 'Simple'
custom_data = False
if model_type == 'Inception3':
    image_size = 299
elif model_type == 'Inception_transfer':
    image_size = 299

transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
transform_array = transforms.Compose([transforms.Resize(image_size), transform_rgb,
                                      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ToTensor()])

if model_type == 'Simple':
    transform_array = transforms.Compose([transforms.Resize(image_size),
                                          # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          ToTensor()])

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_array)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_array)

filenames = []
for root, dir, files in os.walk('data/football'):
    for file in files:
        # print(file)
        if not file.startswith('.'):
            filenames.append(file)

model_name = "model_inceptionv3_test.pth"
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

batch_size = 64
# create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

if custom_data:
    classes = ['player', 'referee', 'ball']

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ToTensor()
        ])
    }

    # CUSTOM DATA
    train_data = FootballDataset('data/football', filenames, transform=data_transforms['train'])
    test_data = FootballDataset('data/football', filenames, transform=data_transforms['test'])

    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=0)

print(train_data)
print(test_data)

# look at batched data
for i_batch, sample_batched in enumerate(test_dataloader):
    print(i_batch, len(sample_batched))
    print(sample_batched[0].shape, sample_batched[1].shape)
    if i_batch == 3:
        break

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]:", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# transfer learning functional
def train_model_functional():
    model_ft = models.inception_v3(pretrained=True, aux_logits=False)
    # model_ft = nn.Sequential(*list(model_ft.children())[:13])
    print(model_ft(torch.randn(1, 3, 299, 299)).shape)
    # print("model:", model_ft)
    num_ftrs = 768
    num_ftrs = model_ft.fc.in_features
    print(num_ftrs)
    model_ft.fc = nn.Linear(num_ftrs, len(classes))
    model_ft = model_ft.to(device)
    return model_ft

is_inception = False
if model_type == 'Simple':
    is_inception = False
    model_ft = NeuralNetwork(classes, image_size).to(device)
elif model_type == 'Inception3':
    is_inception = True
    model_ft = Inception3(num_classes=len(classes)).to(device)
elif model_type == 'Inception_transfer':
    is_inception = False
    model_ft = Inceptionnet(classes, aux_logits=True).to(device)
elif model_type == 'Resnet':
    model_ft = Resnet(classes).to(device)
elif model_type == 'Alexnet':
    model_ft = Alexnet(classes).to(device)
# print("model:", model_ft)
#train model
loss_fn = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1e-3)

# set model: options [model, model_ft]
train_model = model_ft
train_optimizer = optimizer_ft


def train(dataloader, model, loss_fn, optimizer, is_inception=False):
    size = len(dataloader.dataset)
    print("train size", size)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        if is_inception:
            pred, aux_outputs = model(X)
            loss_pred = loss_fn(pred, y)
            loss_aux = loss_fn(aux_outputs, y)
            loss = loss_pred + 0.4 * loss_aux
        else:
            pred = model(X)
            loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: {:7f} [{:5d}/{:5d}]".format(loss, current, size))

    return float(loss)


def test(dataloader, model):
    size = len(dataloader.dataset)
    print("test size", size)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("Test Error: \n Accuracy: {:0.1f}, Avg loss: {:8f}".format(100*correct, test_loss))
    return 100*correct, test_loss


epochs = 1
accuracy = 0
test_loss = 0
train_loss = 0
for t in range(epochs):
    print("Epoch {}\n---------".format(t+1))
    train_loss = train(train_dataloader, train_model, loss_fn, train_optimizer, is_inception=is_inception)
    accuracy, test_loss = test(test_dataloader, train_model)
print("Done!")

train_model.eval()
# x, y = test_data[1][0], test_data[1][1]

with torch.no_grad():
    for i in range(10):
        x, y = test_data[i][0].to(device), test_data[i][1]
        print("x", x.unsqueeze(0).shape, y)
        pred = train_model(x.unsqueeze(0))
        print("pred", pred.shape)
        # pred = torch.max(pred, 1)
        # print(pred)
        predicted = classes[pred[0].argmax(0)]
        actual = classes[y]
        print("Predicted: '{}', Actual: '{}'".format(predicted, actual))


results_json = {'model': model_type,
                'epochs': epochs,
                'train_loss': train_loss,
                'accuracy': accuracy,
                'test loss': test_loss}

# if os.stat('record.json').st_size == 0:
with open('record.json', 'r+') as f:
    data = json.load(f)
    data["Results"].append(results_json)
    f.seek(0)
    json.dump(data, f, indent = 4)

# save model weights
torch.save(train_model.state_dict(), model_name)
print("Saved PyTorch Model State to {}".format(model_name))

# # save entire model
torch.save(train_model, model_name)
model_pred = torch.load(model_name)
model_pred.to("cpu")

# Load state dict for prediction (not quite working)
# model_pred = NeuralNetwork()
# model_pred = models.inception_v3()
# num_ftrs = model_pred.fc.in_features
# model_pred.fc = nn.Linear(num_ftrs, len(classes))
# model_pred.load_state_dict(torch.load(model_name))
# print("loading", model_pred)

model_pred.eval()

with torch.no_grad():
    for i in range(10):
        x, y = test_data[i][0], test_data[i][1]
        pred = model_pred(x.unsqueeze(0))
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print("Reload Model: Predicted: '{}', Actual: '{}'".format(predicted, actual))

