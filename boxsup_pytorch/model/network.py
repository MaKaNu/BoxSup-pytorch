"""Network Module."""
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class FCN8s(nn.Module):
    def __init__(
        self,
        nclass,
        device,
        backbone="vgg16",
        aux=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super(FCN8s, self).__init__()
        self.device = device
        self.aux = aux
        if backbone == "vgg16":
            self.pretrained = vgg16(weights=VGG16_Weights.DEFAULT).features
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))
        self.pool3 = nn.Sequential(*self.pretrained[:17])
        self.pool4 = nn.Sequential(*self.pretrained[17:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool3 = nn.Conv2d(256, nclass, 1)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__(
            "exclusive",
            ["head", "score_pool3", "score_pool4", "auxlayer"]
            if aux
            else ["head", "score_pool3", "score_pool4"],
        )
        self.to(self.device)

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(
            score_fr, score_pool4.size()[2:], mode="bilinear", align_corners=True
        )
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(
            fuse_pool4, score_pool3.size()[2:], mode="bilinear", align_corners=True
        )
        fuse_pool3 = upscore_pool4 + score_pool3

        out = F.interpolate(fuse_pool3, x.size()[2:], mode="bilinear", align_corners=True)

        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode="bilinear", align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        )

    def forward(self, x):
        return self.block(x)
