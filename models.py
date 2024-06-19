import scipy.misc
import scipy.io
from ops import *
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
from time import time


def init_parameters_recursively(layer):
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            init_parameters_recursively(sub_layer)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, std=0.01)
    else:
        return


class ImageNetMI(nn.Module):
    def __init__(self, cfg):
        super(ImageNetMI, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.cnn = nn.Sequential(*list(models.resnet50(pretrained=True).type(torch.float32).children())[:-1])
        self.feature = nn.Sequential(
            nn.Linear(2048, self.SEMANTIC_EMBED * 2),
        )
        self.cross_feature = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.SEMANTIC_EMBED),
        )
        self.hash = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.bit),
            nn.Tanh()
        )
        self.label = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.numClass),
            nn.Sigmoid()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.label)
        init_parameters_recursively(self.cross_feature)

    def forward(self, inputs):
        base = self.cnn(inputs).view(inputs.shape[0], -1)
        mu_sigma_I = self.feature(base)
        mu_I = mu_sigma_I[:, :self.SEMANTIC_EMBED]
        log_sigma_I = mu_sigma_I[:, self.SEMANTIC_EMBED:]
        std_I = log_sigma_I.mul(0.5).exp_()
        fea_I = torch.relu(torch.randn_like(mu_I) * std_I + mu_I)
        fea_T_pred = self.cross_feature(fea_I)
        hsh_I = self.hash(fea_I)
        lab_I = self.label(fea_I)
        return torch.squeeze(fea_I), torch.squeeze(hsh_I), torch.squeeze(lab_I), fea_T_pred, mu_I, log_sigma_I

    def get_hash(self, feature):
        hsh_I = self.hash(feature)
        return torch.squeeze(hsh_I)


class LabelNet(nn.Module):
    def __init__(self, cfg):
        super(LabelNet, self).__init__()
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.bit, kernel_size=(1, self.numClass), stride=(1, 1), bias=False),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.hash)

    def forward(self, inputs):
        hsh_I = self.hash(inputs.view(inputs.shape[0], 1, 1, -1))
        return torch.squeeze(hsh_I)


class TextNetMI(nn.Module):
    def __init__(self, cfg):
        super(TextNetMI, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.dimTxt = cfg.dimTxt
        self.interp_block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 1 * 5], stride=[1, 1 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 2 * 5], stride=[1, 2 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 3 * 5], stride=[1, 3 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 6 * 5], stride=[1, 6 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block10 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 10 * 5], stride=[1, 10 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=4096, kernel_size=(1, self.dimTxt), stride=(1, 1)),
            nn.ReLU(),
            nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0),
            nn.Conv2d(in_channels=4096, out_channels=2 * self.SEMANTIC_EMBED, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
        )
        self.cross_feature = nn.Sequential(
            nn.Conv2d(in_channels=self.SEMANTIC_EMBED, out_channels=self.SEMANTIC_EMBED, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.norm = nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0)
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=self.SEMANTIC_EMBED, out_channels=self.bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh(),
        )
        self.label = nn.Sequential(
            nn.Conv2d(in_channels=self.SEMANTIC_EMBED, out_channels=self.numClass, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.interp_block1)
        init_parameters_recursively(self.interp_block2)
        init_parameters_recursively(self.interp_block3)
        init_parameters_recursively(self.interp_block6)
        init_parameters_recursively(self.interp_block10)
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.cross_feature)
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.label)

    def forward(self, inputs):
        unsqueezed = inputs.view(inputs.shape[0], 1, 1, -1)
        interp_in1 = F.interpolate(self.interp_block1(unsqueezed), size=(1, self.dimTxt))
        interp_in2 = F.interpolate(self.interp_block2(unsqueezed), size=(1, self.dimTxt))
        interp_in3 = F.interpolate(self.interp_block3(unsqueezed), size=(1, self.dimTxt))
        interp_in6 = F.interpolate(self.interp_block6(unsqueezed), size=(1, self.dimTxt))
        interp_in10 = F.interpolate(self.interp_block10(unsqueezed), size=(1, self.dimTxt))
        MultiScal = torch.cat([
            unsqueezed,
            interp_in10,
            interp_in6,
            interp_in3,
            interp_in2,
            interp_in1
        ], 1)
        mu_sigma_T = self.feature(MultiScal)
        mu_T = mu_sigma_T[:, :self.SEMANTIC_EMBED]
        log_sigma_T = mu_sigma_T[:, self.SEMANTIC_EMBED:]
        std_T = log_sigma_T.mul(0.5).exp_()
        fea_T = torch.relu(torch.randn_like(mu_T) * std_T + mu_T)
        fea_I_pred = self.cross_feature(fea_T)
        norm = self.norm(fea_T)
        hsh_T = self.hash(norm)
        lab_T = self.label(norm)
        tuple = torch.squeeze(fea_T), torch.squeeze(hsh_T), torch.squeeze(lab_T), torch.squeeze(fea_I_pred), torch.squeeze(mu_T), torch.squeeze(log_sigma_T)
        return tuple

    def get_hash(self, feature):
        fea_T = feature.reshape([feature.shape[0], feature.shape[1], 1, 1])
        norm = self.norm(fea_T)
        hsh_T = self.hash(norm)
        return torch.squeeze(hsh_T)


class ImageNetV0(nn.Module):
    def __init__(self, cfg):
        super(ImageNetV0, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.cnn = nn.Sequential(*list(models.resnet50(pretrained=True).type(torch.float32).children())[:-1])
        self.feature = nn.Sequential(
            nn.Linear(2048, self.SEMANTIC_EMBED),
        )
        self.hash = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.bit),
            nn.Tanh()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)

    def forward(self, inputs):
        base = self.cnn(inputs).view(inputs.shape[0], -1)
        fea_I = self.feature(base)
        hsh_I = self.hash(fea_I)
        return torch.squeeze(hsh_I)


class TextNetV0(nn.Module):
    def __init__(self, cfg):
        super(TextNetV0, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.dimTxt = cfg.dimTxt
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4096, kernel_size=(1, self.dimTxt), stride=(1, 1)),
            nn.ReLU(),
            nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0),
            nn.Conv2d(in_channels=4096, out_channels=self.SEMANTIC_EMBED, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
        )
        self.norm = nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0)
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=self.SEMANTIC_EMBED, out_channels=self.bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)

    def forward(self, y):
        # y.shape == (N, 1, 1, LEN)
        unsqueezed = y.view(y.shape[0], 1, 1, -1)
        fea_T = self.feature(unsqueezed)
        norm = self.norm(fea_T)
        hash = self.hash(norm)
        return torch.squeeze(hash)


class ImageNet(nn.Module):
    def __init__(self, cfg):
        super(ImageNet, self).__init__()
        self.bit = cfg.bit
        self.pre_feature = nn.Sequential(*list(models.resnet50(pretrained=True).type(torch.float32).children())[:-1])
        self.hash = nn.Sequential(
            nn.Linear(2048, self.bit, bias=False),
            nn.Tanh()
        )
        self.cross_feature = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=8192, kernel_size=(1, 1),
                      stride=(1, 1)),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.latent)

    def forward(self, x):
        pre_feature = self.pre_feature(x)
        pre_feature = pre_feature.view(pre_feature.shape[0], -1)
        hash = self.hash(pre_feature)
        latent = self.latent(pre_feature)
        return torch.squeeze(hash), torch.squeeze(latent)


class TextNet(nn.Module):
    # y.shape == (N, 1, 1, LEN)
    def __init__(self, cfg):
        super(TextNet, self).__init__()
        self.bit = cfg.bit
        self.dimTxt = cfg.dimTxt
        self.pre_feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8192, kernel_size=(1, self.dimTxt), stride=(1, 1)),
            nn.ReLU(),
        )
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=8192, out_channels=self.bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh()
        )
        self.cross_feature = nn.Sequential(
            nn.Conv2d(in_channels=8192, out_channels=2048, kernel_size=(1, 1),
                      stride=(1, 1)),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.pre_feature)
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.latent)

    def forward(self, y):
        # y.shape == (N, 1, 1, LEN)
        unsqueezed = y.view(y.shape[0], 1, 1, -1)
        pre_feature = self.pre_feature(unsqueezed)
        hash = self.hash(pre_feature)
        latent = self.latent(pre_feature)
        mu_sigma_T = self.feature(MultiScal)
        mu_T = mu_sigma_T[:, :self.SEMANTIC_EMBED]
        log_sigma_T = mu_sigma_T[:, self.SEMANTIC_EMBED:]
        std_T = log_sigma_T.mul(0.5).exp_()
        fea_T = torch.relu(torch.randn_like(mu_T) * std_T + mu_T)
        fea_I_pred = self.cross_feature(fea_T)
        norm = self.norm(fea_T)
        hsh_T = self.hash(norm)
        lab_T = self.label(norm)
        tuple = torch.squeeze(fea_T), torch.squeeze(hsh_T), torch.squeeze(lab_T), torch.squeeze(
            fea_I_pred), torch.squeeze(mu_T), torch.squeeze(log_sigma_T)
        return tuple
        return torch.squeeze(hash), torch.squeeze(latent)
