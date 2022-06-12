from torch import nn
import torch.nn.functional as F
from .utils import make_divisible


class ConvBNAct(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size = (3, 3),
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups = 1,
        norm_layer = nn.BatchNorm2d,
        act_layer = nn.ReLU6
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        if norm_layer == nn.Identity:
            self.bn = norm_layer()
        else:
            self.bn = norm_layer(out_channels)
            
        self.act = act_layer(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, expansion_ratio: float = 6):
        super().__init__()
        
        expansion_channels = make_divisible(in_channels * expansion_ratio)
        self.residual = in_channels == out_channels and stride == 1
        
        if in_channels != expansion_channels:
            self.expand = ConvBNAct(
                in_channels=in_channels,
                out_channels=expansion_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            )
        else:
            self.expand = nn.Identity()
        
        self.depthwise = ConvBNAct(
            in_channels=expansion_channels,
            out_channels=expansion_channels,
            stride=stride,
            kernel_size=(3, 3),
            padding=1,
            groups=expansion_channels,
            bias=False,
        )
        self.pointwise = ConvBNAct(
            in_channels=expansion_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=0,
            bias=False,
            act_layer=nn.Identity
        )
        
    def forward(self, x):
        y = self.expand(x)
        y = self.depthwise(y)
        y = self.pointwise(y)
        if self.residual:
            return x + y
        else:
            return y