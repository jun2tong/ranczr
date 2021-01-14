import copy
import torch
import torch.nn as nn
import timm

from collections import OrderedDict
from torch.nn import functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from efficientnet_pytorch import EfficientNet


class CustomResNext(nn.Module):
    def __init__(self, model_name, pretrained=True, target_size=11):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained, progress=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomXception(nn.Module):
    def __init__(self, pretrained_path, target_size=11):
        super().__init__()
        self.model = timm.models.xception(pretrained=True)
        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(checkpoint)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomEffNet(nn.Module):
    def __init__(self, model_name, target_size=11):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained(model_name)
        # self.effnet = timm.create_model(model_name,pretrained=True,num_classes=target_size)

        self.effnet._dropout = nn.Dropout(0.1)
        n_features = self.effnet._fc.in_features
        self.effnet._fc = nn.Linear(n_features, target_size)

    def forward(self, image, targets=None):
        outputs = self.effnet(image)
        return outputs
