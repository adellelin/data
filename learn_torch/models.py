import torch
from torch import nn
from torchvision import datasets, transforms, models

# simple model
class NeuralNetwork(nn.Module):
    def __init__(self, classes, image_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.classes = classes
        self.image_size = image_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(image_size*image_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(classes)),
            nn.ReLU()
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.linear_relu_stack(x)
        return logits


class Alexnet(nn.Module):
    def __init__(self, classes):
        super(Alexnet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.classes = classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5, inplace=False),
            nn.Linear(9216, 4096, True),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, 4096, True),
            nn.ReLU(),
            nn.Linear(4096, len(self.classes), True)
        )

    def forward(self, x):
        x = self.model(x)  #['InceptionOutputs']
        return x


class Inceptionnet(nn.Module):
    def __init__(self, classes, aux_logits=False):
        super(Inceptionnet, self).__init__()
        self.aux_logits = aux_logits
        model = models.inception_v3(pretrained=True, aux_logits=self.aux_logits)
        modules = list(model.children())[:16]
        self.model = nn.Sequential(*modules)
        self.classes = classes
        self.fc = nn.Sequential(
            nn.Linear(1280, len(self.classes), True)
        )
        if self.aux_logits:
            self.model[15].fc = nn.Linear(768, len(self.classes))

    def forward(self, x):
        x = self.model(x)
        if not self.aux_logits:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            # print(x.size())
            x = self.fc(x)
        return x


class Resnet(nn.Module):
    def __init__(self, classes):
        super(Resnet, self).__init__()
        model = models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.classes = classes
        self.fc = nn.Sequential(
            nn.Linear(512, len(self.classes))
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(f.size(0), -1)
        # print(x.size())
        x = self.fc(f)
        return x