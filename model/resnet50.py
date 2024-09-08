from torch import nn
import torch
import torchvision.models as models


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

    def forward(self, x):
        return x


class Resnet50(nn.Module):
    def __init__(self, require_grad=False):
        super(Resnet50, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = FC()

        if require_grad:
            for i, param in enumerate(self.resnet.parameters()):
                if i < 129:
                    param.requires_grad = False
        else:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 2))

    def forward(self, res_input):
        x = self.resnet(res_input)
        x = self.classifier(x)
        return x