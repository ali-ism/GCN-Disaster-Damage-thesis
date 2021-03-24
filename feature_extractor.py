import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
import copy
from twostream_resnet50_diff import Bridge
from download import download_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FeatureExtractor(torch.nn.Module):
    def __init__(self,
        pretrained = False,
        shared = False,
        diff = True
        ):
        """
        Builds the full ResNet50 Diff model then truncated down to the Bridge.

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

    def forward(self, x):
        """
        Same Forward pass as the full ResNet50 Diff model but stops at the Bridge output.
        
        Args:
            x (torch.Tensor): The 6-channel input
        
        Returns:
            torch.Tensor: A 2048x8x8 output feature map
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
        return self.bridge(x)


def loadFeatureExtractor(setting_path = './resnet50_settings.json'):
    """
        Loads the ResNet50 feature extractor.
        
        Args:
            setting_path (str): Path to the model setting JSON file
        
        Returns:
            model (torch.nn.Module): The feature extraction model
            
    """
    with open(setting_path, 'r') as JSON:
        setting_dict = json.load(JSON)
    model_args = setting_dict["model"]["model_args"]
    pretrained = model_args['pretrained']
    shared = model_args['shared']
    diff = model_args['diff']
    model = FeatureExtractor(pretrained, shared, diff)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if not pretrained:
        weight_path = download_weights('table_1_plain')
        model.load_state_dict(torch.load(weight_path)["state_dict"], strict=False) #TODO test this
    model.to(device)
    return model