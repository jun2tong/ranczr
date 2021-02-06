import copy
import torch
import torch.nn as nn
import timm
import pdb

from collections import OrderedDict
from torch.nn import functional as F
from torch.cuda.amp import autocast
from efficientnet_pytorch import EfficientNet
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
    def __init__(self, model_name="resnet200d", out_dim=11):
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


class MyEnsemble(nn.Module):
    def __init__(self, weight_paths):
        super(MyEnsemble, self).__init__()
        self.model_lst = nn.ModuleList()
        for each in weight_paths:
            model = RANCZRResNet200D()
            model.load_state_dict(torch.load(each, map_location="cpu"))
            for param in model.parameters():
                param.requires_grad = False
            self.model_lst.append(model)

    def forward(self, x):
        preds = []
        for mod in self.model_lst:
            preds.append(mod(x).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        preds = torch.mean(preds, dim=0)
        return preds

class CustomAttention(nn.Module):
    
    def __init__(self, model_name, target_size, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(4,))
        self.num_feas = self.backbone.feature_info.channels()[-1]

        self.local_fe = CBAM(self.num_feas)
        self.dropout = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.num_feas, target_size)

        # self.classifier = nn.Sequential(nn.Linear(self.num_feas+self.local_fe.inter_channels, self.num_feas),
        #                                 # nn.BatchNorm1d(self.num_feas),
        #                                 # nn.Dropout(0.2),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.num_feas, target_size))

    def forward(self, x):
        feas = self.backbone(x)[0]
        # glob_feas = self.global_pool(feas)
        # glob_feas = self.dropout(glob_feas.flatten(start_dim=1))

        all_feas = self.local_fe(feas)
        all_feas = self.global_pool(all_feas)
        all_feas = all_feas.flatten(start_dim=1)
        all_feas = self.dropout(all_feas)
        # local_feas = self.dropout(torch.sum(local_feas, dim=[2,3]))

        # all_feas = torch.cat([glob_feas, local_feas], dim=1)
        outputs = self.classifier(all_feas)
        return outputs

class EffNetWLF(nn.Module):

    def __init__(self, model_name, target_size=11):
        super().__init__()
        self.backbone = EfficientNet.from_name(model_name)

        self.backbone._dropout = nn.Dropout(0.1)
        n_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(n_features, target_size)

        # self.backbone._fc = nn.Identity()
        self.local_fe = CBAM(n_features)
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(n_features, target_size)

        # self.classifier = nn.Sequential(nn.Linear(n_features+self.local_fe.inter_channels, n_features),
        #                                 nn.ReLU(),
        #                                 nn.Linear(n_features, target_size))

    def forward(self, image):
        enc_feas = self.backbone.extract_features(image)

        # use default's global features
        global_feas = self.local_fe(enc_feas)
        global_feas = self.backbone._avg_pooling(global_feas)
        global_feas = global_feas.flatten(start_dim=1)
        global_feas = self.dropout(global_feas)

        # local features
        # local_feas = self.local_fe(enc_feas)
        # local_feas = torch.sum(local_feas, dim=[2,3])
        # local_feas = self.dropout(local_feas)

        # all_feas = torch.cat([global_feas, local_feas], dim=1)
        outputs = self.classifier(global_feas)
        return outputs


class CustomModel(nn.Module):
    
    def __init__(self, model_name, target_size):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(4,))
        self.num_feas = self.backbone.feature_info.channels()[-1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_feas, target_size)

    def forward(self, x):
        feas = self.backbone(x)[0]

        all_feas = self.global_pool(feas)
        all_feas = all_feas.flatten(start_dim=1)
        outputs = self.fc(all_feas)
        return outputs
