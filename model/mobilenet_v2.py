import torch
from torch import nn
import torch.nn.functional as F

from .blocks import InvertedResidual, ConvBNAct
from .utils import apply_width_mult


def repeated_blocks(cls = InvertedResidual, n=1, **kwargs):
    blocks = []
    
    for layer_idx in range(n):
        is_first_layer = layer_idx == 0
        if not is_first_layer:
            kwargs['in_channels'] = kwargs['out_channels']  # after first layer, in_channels is equal to out_channels
        if not is_first_layer:
            kwargs['stride'] = 1  # stride only applies to first layer
        blocks.append(cls(**kwargs))
    return nn.Sequential(*blocks)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        
        # Shorthand operator
        _apply = lambda in_channel: apply_width_mult(in_channel, width_mult)
        
        self.conv1 = ConvBNAct(
            3, _apply(32), (3, 3), stride=2, bias=False
        )
        
        self.stage1 = repeated_blocks(n=1, in_channels=_apply(32), out_channels=_apply(16), expansion_ratio=1)  # expansion_ratio defaults to 6
        self.stage2 = repeated_blocks(n=2, in_channels=_apply(16), out_channels=_apply(24), stride=2)
        self.stage3 = repeated_blocks(n=3, in_channels=_apply(24), out_channels=_apply(32), stride=2)
        self.stage4 = repeated_blocks(n=4, in_channels=_apply(32), out_channels=_apply(64), stride=2)
        self.stage5 = repeated_blocks(n=3, in_channels=_apply(64), out_channels=_apply(96), stride=1)
        self.stage6 = repeated_blocks(n=3, in_channels=_apply(96), out_channels=_apply(160), stride=2)
        self.stage7 = repeated_blocks(n=1, in_channels=_apply(160), out_channels=_apply(320), stride=1)
        
        self.conv2 = ConvBNAct(
            _apply(320), _apply(1280), (1, 1), stride=1, padding=0, bias=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(_apply(1280), num_classes, bias=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    # Test code
    from torchinfo import summary
    
    for resolution in [96, 128, 160, 192, 224]:
        wm_list = [0.35, 0.5, 0.75, 1.0]
        if resolution == 224:
            wm_list.append(1.4)
        for width_mult in wm_list:
            print("width_mult={} {}x{} ... ".format(width_mult, resolution, resolution), end='', flush=True)
            _ = summary(MobileNetV2(num_classes=1000, width_mult=1.0), (1, 3, 224, 224), verbose=False)
            print("PASS")