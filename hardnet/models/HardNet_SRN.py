# CNN model definition package

import torch
import torch.nn as nn
import torch.nn.functional as F
from hardnet.Utils import L2Norm
from collections import OrderedDict

# Weights were initialized to orthogonally with gain equal to 0.6,
# biases set to 0.01 Training
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return


class HardNet_SRN(nn.Module):
    """HardNet model definition with arf/srn/arf+srn
    """

    def __init__(self, use_smooth=False):
        super(HardNet_SRN, self).__init__()
        self.use_smooth = use_smooth

        # network definition
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(32, affine=False)),
            ('conv2', nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(32, affine=False)),

            ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(64, affine=False)),
            ('conv4', nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(64, affine=False)),

            ('conv5', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn5', nn.BatchNorm2d(128, affine=False)),
            ('conv6', nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)),
            ('bn6', nn.BatchNorm2d(128, affine=False)),

            ('fc', nn.Conv2d(128, 128, kernel_size=8, bias=False)),
            ('bn_fc', nn.BatchNorm2d(128, affine=False))
        ]))

        # Lateral layers
        self.latlayer1 = nn.Conv2d( 64, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 32, 128, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.features.apply(weights_init)
        self.latlayer1.apply(weights_init)
        self.latlayer2.apply(weights_init)
        self.smooth.apply(weights_init)


    # pre-processing for the input data
    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable)   feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    # return the L2Norm regularization
    def forward(self, input):
        # Bottom-up
        c1 = F.relu(self.features.bn1(self.features.conv1(self.input_norm(input))))
        c2 = F.relu(self.features.bn2(self.features.conv2(c1)))
        c3 = F.relu(self.features.bn3(self.features.conv3(c2)))
        c4 = F.relu(self.features.bn4(self.features.conv4(c3)))
        c5 = F.relu(self.features.bn5(self.features.conv5(c4)))
        c6 = F.relu(self.features.bn6(self.features.conv6(c5)))
        # Top-down

        p1 = F.max_pool2d(self.latlayer1(c4), 2) + c6 #c6+c4
        p2 = F.max_pool2d(self.latlayer2(c2), 4) + p1 #c6+c4+c2
        # Smooth
        if self.use_smooth:
            p2 = self.smooth(p2)

        x_features = self.features.bn_fc(self.features.fc(F.dropout(p2, 0.3)))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)