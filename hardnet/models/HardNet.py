# CNN model definition package

import torch
import torch.nn as nn
import torch.nn.functional as F
from hardnet.Utils import L2Norm
from orn.modules import ORConv2d

# Weights were initialized to orthogonally with gain equal to 0.6,
# biases set to 0.01 Training
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, ORConv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return


class HardNet(nn.Module):
    """HardNet model definition with arf/srn/arf+srn
    """

    def __init__(self, use_arf=False, nOrientation=8):
        super(HardNet, self).__init__()
        if nOrientation==4 or nOrientation==8:
            self.nOrientation = nOrientation
        else:
            self.nOrientation = 8

        self.use_arf = use_arf
        if self.use_arf:
            self.features = nn.Sequential(
                ORConv2d(1, 32/nOrientation, arf_config=(1, nOrientation), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=False),
                nn.ReLU(),
                ORConv2d(32/nOrientation, 32/nOrientation, arf_config=nOrientation, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),#output=16

                ORConv2d(32/nOrientation, 64/nOrientation, arf_config=nOrientation, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=False),
                nn.ReLU(),
                ORConv2d(64/nOrientation, 64/nOrientation, arf_config=nOrientation, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # output=8

                ORConv2d(64/nOrientation, 128/nOrientation, arf_config=nOrientation, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # output=4

                ORConv2d(128/nOrientation, 128/nOrientation, arf_config=nOrientation, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4),  # output=1
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=False), # stable params in BN
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=False),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=False),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=False),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=False),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=False),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv2d(128, 128, kernel_size=8, bias=False),
                nn.BatchNorm2d(128, affine=False),
            )

        self.features.apply(weights_init)

    # pre-processing for the input data
    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    # return the L2Norm regularization
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)