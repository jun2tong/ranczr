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
from custom_mod.inception import inception_v3
from custom_mod.bit_resnet import ResNetV2, get_weights


class CustomResNext(nn.Module):
    def __init__(self, model_name, target_size=11):
        super().__init__()
        weights = get_weights(f"pre-trained/{model_name}.npz")
        # ("BiT-M-R50x1", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        # ("BiT-M-R50x3", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        # ("BiT-M-R101x1", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        # ("BiT-M-R101x3", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),

        self.model = ResNetV2([3, 4, 6, 3], 3, head_size=target_size, zero_head=True)
        self.model.load_from(weights)
        print("Load complete")

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


class CustomInceptionV3(nn.Module):
    def __init__(self, target_size=11):
        super().__init__()
        self.model = inception_v3(pretrained=True)
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
