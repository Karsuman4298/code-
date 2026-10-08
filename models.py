import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    return x.view(b, groups, c // groups, h, w).transpose(1, 2).reshape(b, c, h, w)


def init_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class MultiScaleConv(nn.Module):
    def __init__(self, channels, kernels=(1, 3, 5), expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.groups = math.gcd(hidden, channels)
        self.expand = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True))
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)) for k in kernels]
        )
        self.project = nn.Sequential(nn.Conv2d(hidden, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        h = self.expand(x)
        y = self.branches[0](h)
        for branch in self.branches[1:]:
            y = y + branch(h)
        return x + self.project(channel_shuffle(y, self.groups))


class ChannelGate(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        reduced = max(1, channels // min(ratio, channels))
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=False)

    def forward(self, x):
        avg = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        mx = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        return torch.sigmoid(avg + mx)


class SpatialGate(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class UpConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.proj = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.proj(self.up(x))


class AttentionGate(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        mid = channels // 2
        self.wg = nn.Sequential(nn.Conv2d(channels, mid, kernel, padding=kernel // 2, groups=mid), nn.BatchNorm2d(mid))
        self.wx = nn.Sequential(nn.Conv2d(channels, mid, kernel, padding=kernel // 2, groups=mid), nn.BatchNorm2d(mid))
        self.psi = nn.Sequential(nn.Conv2d(mid, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))


class EMCADDecoder(nn.Module):
    def __init__(self, channels, kernels=(1, 3, 5), expansion=2):
        super().__init__()
        c4, c3, c2, c1 = channels
        self.spatial = SpatialGate()
        self.gate = nn.ModuleList([ChannelGate(c) for c in channels])
        self.conv = nn.ModuleList([MultiScaleConv(c, kernels, expansion) for c in channels])
        self.up = nn.ModuleList([UpConv(c4, c3), UpConv(c3, c2), UpConv(c2, c1)])
        self.att = nn.ModuleList([AttentionGate(c3), AttentionGate(c2), AttentionGate(c1)])
        init_decoder(self)

    def refine(self, i, x):
        x = self.gate[i](x) * x
        x = self.spatial(x) * x
        return self.conv[i](x)

    def forward(self, f4, skips):
        outs = [self.refine(0, f4)]
        for i, skip in enumerate(skips):
            d = self.up[i](outs[-1])
            outs.append(self.refine(i + 1, d + self.att[i](d, skip)))
        return outs


class EMCADNet(nn.Module):
    def __init__(self, pretrained=True, expansion=2):
        super().__init__()
        self.encoder = timm.create_model("pvt_v2_b2", pretrained=pretrained, features_only=True)
        channels = list(reversed(self.encoder.feature_info.channels()))
        self.decoder = EMCADDecoder(channels, expansion=expansion)
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in channels])

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        outs = self.decoder(f4, [f3, f2, f1])
        size = x.shape[-2:]
        return [F.interpolate(head(d), size=size, mode="bilinear", align_corners=False) for head, d in zip(self.heads, outs)]


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class UNetR50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model("resnet50", pretrained=pretrained, features_only=True)
        c = self.encoder.feature_info.channels()
        d = [256, 128, 64, 32, 16]
        self.blocks = nn.ModuleList(
            [DecoderBlock(c[4], c[3], d[0]), DecoderBlock(d[0], c[2], d[1]), DecoderBlock(d[1], c[1], d[2]), DecoderBlock(d[2], c[0], d[3]), DecoderBlock(d[3], 0, d[4])]
        )
        self.head = nn.Conv2d(d[4], 1, 3, padding=1)

    def forward(self, x):
        f = self.encoder(x)
        y = f[4]
        for block, skip in zip(self.blocks, [f[3], f[2], f[1], f[0], None]):
            y = block(y, skip)
        return [self.head(y)]


def build_model(name, pretrained=True):
    if name == "emcad_b2":
        return EMCADNet(pretrained=pretrained)
    if name == "unet_r50":
        return UNetR50(pretrained=pretrained)
    raise SystemExit(f"unknown model {name}")
