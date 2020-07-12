"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as models_2d 

model_dict = {
    'densenet121': [models_2d.densenet121, 1024],
    'densenet161': [models_2d.densenet161, 2208],
    'densenet169': [models_2d.densenet169, 1664],
    'densenet201': [models_2d.densenet201, 1920]
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class CheXpert(nn.Module):
    """Normal DenseNet for CheXpert"""
    def __init__(self, model_name='densenet121', num_classes=14, ckpt_path=None):
        super(CheXpert, self).__init__()

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            opt = ckpt['opt']
            model_state = ckpt['model']

            # handle models weights saved in dataparallel
            if "module" in list(model_state.keys())[0]:
                temp_model_state = {}
                for k,v in model_state.items():
                    new_k = ".".join(k.split(".module."))
                    temp_model_state[new_k] = v
                model_state = temp_model_state 

            self.model = SupConDenseNet(name=opt.model)
            self.model.load_state_dict(model_state)
            dim_in = self.model.dim_in
        else:
            model_fun, dim_in = model_dict[model_name]
            self.model= model_fun()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.model.classifier = nn.Linear(dim_in, num_classes)
        self.model_name = model_name 
        self.num_classes = num_classes


    def forward(self, x):

        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.classifier(x)
        return x

    def args_dict(self):
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes
        }


class SupConDenseNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='densenet121', head='mlp', feat_dim=128):
        super(SupConDenseNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.dim_in = dim_in
        self.encoder = model_fun()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.module.features(x) # batch x feature_size x 3 x 3
        #feat = self.encoder.features(x) # batch x feature_size x 3 x 3
        feat = F.relu(feat, inplace=True)
        feat = self.pool(feat) 
        feat = torch.flatten(feat, 1)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='densenet121', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='densenet121', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


if __name__ == "__main__":
    # test
    ckpt_path = "/data4/selfsupervision/log/SupCon/chexpert_models/SupCon_chexpert_densenet161_lr_0.0006752719463409214_decay_0.0001_bsz_4_temp_0.07_trial_0/last.pth" 
    model = CheXpert(ckpt_path=ckpt_path)