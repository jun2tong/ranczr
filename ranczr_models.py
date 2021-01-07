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


class SegModel(nn.Module):
    # Use this with lung mask data
    
    def __init__(self, backbone_name, target_size=11, out_indices=None):
        super().__init__()
        if backbone_name == 'resnet50':
            self.backbone = timm.create_model(backbone_name, features_only=True, pretrained=True, out_indices=out_indices)
            print(f"created {backbone_name} backbone with {backbone.feature_info.channels()} features")
            # self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=13)
        else:
            self.backbone = EfficientNet.from_pretrained(backbone_name)
            print(f"created {backbone_name} backbone")

        # return_layers = {'layer4': 'out',
        #                  'layer3': 'aux'}

        # self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # modify segmentation classifier head
        # self.segment_head = DeepLabHead(2048, 1)
        self.segment_head = DeepLabHead(1408, 13)

        # TODO: modify self.model.classifier
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1408, 2048, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024, target_size)

    def forward(self, x, train=True):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone.extract_features(x)

        result = OrderedDict()
        # seg_feas = features["out"]
        if train:
            # segmentation
            seg_feas = self.segment_head(features)
            seg_feas = F.interpolate(seg_feas, size=input_shape, mode='bilinear', align_corners=False)
            result["seg_out"] = seg_feas

        # classification
        # aux = features["aux"]
        aux = self.relu(self.bn1(self.conv1(features)))
        aux = self.bn2(self.conv2(aux))
        aux = self.avgpool(aux)
        aux = torch.flatten(aux,1)
        aux = self.fc(aux)
        result["cls"] = aux
        return result
