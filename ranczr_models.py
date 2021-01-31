import copy
import torch
import torch.nn as nn
import timm
import pdb

from collections import OrderedDict
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from efficientnet_pytorch import EfficientNet
from custom_mod.inception import inception_v3
from custom_mod.bit_resnet import ResNetV2, get_weights
from custom_mod.attention import CBAM
import segmentation_models_pytorch as smp


class CustomResNext(nn.Module):
    def __init__(self, model_name, target_size=11):
        super().__init__()
        weights = get_weights(f"pre-trained/{model_name}.npz")
        # ("BiT-M-R50x1", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        # ("BiT-M-R50x3", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        # ("BiT-M-R101x1", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        # ("BiT-M-R101x3", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        self.model = ResNetV2([3, 4, 6, 3], 1, head_size=target_size, zero_head=True)
        self.model.load_from(weights)
        # print("Load complete")
        # self.model = timm.models.resnet.ecaresnet50d_pruned(True)
        # n_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(n_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomXception(nn.Module):
    def __init__(self, pretrained_path, target_size=11):
        super().__init__()
        self.model = timm.models.xception(pretrained=False)
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


class SMPModel(nn.Module):
    def __init__(self, model_name, aux_dict, weight_dir):
        super().__init__()
        self.model = smp.Unet(model_name, classes=1, aux_params=aux_dict)
        self.model.load_state_dict(torch.load(weight_dir)["model"])

    def forward(self, x):
        enc_out = self.model.encoder(x)
        out = self.model.classification_head(enc_out[-1])
        return out


class RANCZRResNet200D(nn.Module):
    def __init__(self, model_name="resnet200d", out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


class EffNetWLF(nn.Module):

    def __init__(self, model_name, target_size=11, prev_weights=None):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(model_name)

        self.backbone._dropout = nn.Dropout(0.1)
        n_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(n_features, target_size)

        if prev_weights:
            self.load_from_pth(prev_weights)
            print("loaded previous weights")

        self.local_fe = CBAM(n_features)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(nn.Linear(n_features + n_features, n_features),
                                        nn.BatchNorm1d(n_features),
                                        nn.Dropout(0.1),
                                        nn.ReLU(),
                                        nn.Linear(n_features, target_size))

    def load_from_pth(self, weight_dict):
        new_state_dict = OrderedDict()

        for old_name, val in weight_dict.items():
            new_lst = old_name.split(".")[2:]
            new_name = ".".join(new_lst)
            new_state_dict[new_name] = val
        
        self.backbone.load_state_dict(new_state_dict)

    def forward(self, image):
        enc_feas = self.backbone.extract_features(image)

        # use default's global features
        global_feas = self.backbone._avg_pooling(enc_feas)
        global_feas = global_feas.flatten(start_dim=1)
        global_feas = self.dropout(global_feas)

        local_feas = self.local_fe(enc_feas)
        local_feas = torch.sum(local_feas, dim=[2,3])
        local_feas = self.dropout(local_feas)

        all_feas = torch.cat([global_feas, local_feas], dim=1)
        outputs = self.classifier(all_feas)
        return outputs
