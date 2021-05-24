# ------------------------------------------------------------------------------
# This code is (modified) from
# Assessing out-of-domain generalization for robust building damage detection
# https://github.com/ecker-lab/robust-bdd.git
# Licensed under the CC BY-NC-SA 4.0 License.
# Written by Vitus Benson (vbenson@bgc-jena.mpg.de)
# ------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import copy
from download import download_weights


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class FeatureExtractor(torch.nn.Module):
    def __init__(self,
        pretrained = False,
        shared = False,
        diff = True
        ) -> None:
        """
        Builds the encoder part of the Twostream ResNet50 Diff model (Input to the Bridge module).

        Args:
            pretrained (bool, optional): If True uses ImageNet pretrained weights for the two ResNet50 encoders. Defaults to False.
            shared (bool, optional): If True, model is siamese (shared encoder). Defaults to False.
            diff (bool, optional): If True, difference is fed to decoder, else an intermediate 1x1 conv merges the two downstream features. Defaults to True.
        """
        super(FeatureExtractor, self).__init__()

        self.resnet = resnet50(pretrained=pretrained)
        self.shared = shared
        self.diff = diff
        down_blocks1 = []
        self.input_block1 = copy.deepcopy(nn.Sequential(*list(self.resnet.children()))[:3])
        if not shared:
            down_blocks2 = []
            self.input_block2 = copy.deepcopy(nn.Sequential(*list(self.resnet.children()))[:3])
        self.input_pool = copy.deepcopy(list(self.resnet.children())[3])
        for bottleneck in list(self.resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks1.append(copy.deepcopy(bottleneck))
                if not shared:
                    down_blocks2.append(copy.deepcopy(bottleneck))
        self.down_blocks1 = nn.ModuleList(down_blocks1)
        if not shared:
            self.down_blocks2 = nn.ModuleList(down_blocks2)
        self.bridge = Bridge(2048, 2048)

        if not diff:
            self.concat_blocks = nn.ModuleList([nn.Conv2d(n, int(n/2), kernel_size=1, stride = 1) for n in [4096,2048,1024,512,128,6]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same Forward pass as the full ResNet50 Diff model but stops at the Bridge output.
        
        Args:
            x (torch.Tensor): The 6-channel input
        
        Returns:
            torch.Tensor: A (131072) size output feature vector
        """        
        x1, x2 = torch.split(x, 3, dim = 1)
        del x
        pre_pools1 = dict()
        pre_pools1[f"layer_0"] = x1
        x1 = self.input_block1(x1)
        pre_pools1[f"layer_1"] = x1
        x1 = self.input_pool(x1)

        for i, block in enumerate(self.down_blocks1, 2):
            x1 = block(x1)
            if i == (5):
                continue
            pre_pools1[f"layer_{i}"] = x1

        pre_pools2 = dict()
        pre_pools2[f"layer_0"] = x2
        if not self.shared:
            x2 = self.input_block2(x2)
        else:
            x2 = self.input_block1(x2)
        pre_pools2[f"layer_1"] = x2
        x2 = self.input_pool(x2)

        if not self.shared:
            tmp_down = self.down_blocks2
        else:
            tmp_down = self.down_blocks1
        for i, block in enumerate(tmp_down, 2):
            x2 = block(x2)
            if i == (5):
                continue
            pre_pools2[f"layer_{i}"] = x2

        if self.diff:
            x = torch.add(input=x1, other=x2, alpha=-1)
        else:
            x = self.concat_blocks[0](torch.cat([x1,x2],1))
        return self.bridge(x).flatten()


def load_feature_extractor(pretrained=False, shared=False, diff=True, weight_path=None) -> torch.nn.Module:
    """
        Loads the ResNet50 feature extractor.
        
        Args:
            pretrained (bool, optional): If True uses ImageNet pretrained weights for the two ResNet50 encoders. Defaults to False.
            shared (bool, optional): If True, model is siamese (shared encoder). Defaults to False.
            diff (bool, optional): If True, difference is fed to decoder, else an intermediate 1x1 conv merges the two downstream features. Defaults to True.
        
        Returns:
            model (torch.nn.Module): The feature extraction model
            
    """
    model = FeatureExtractor(pretrained, shared, diff)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if not pretrained:
        if weight_path is None and not os.path.isfile('./weights/twostream-resnet50_all_plain.pt'):
            weight_path = download_weights('table_1_plain')
        elif weight_path is None and os.path.isfile('./weights/twostream-resnet50_all_plain.pt'):
            weight_path = './weights/twostream-resnet50_all_plain.pt'
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weight_path,map_location='cpu')["state_dict"], strict=False)
    return model