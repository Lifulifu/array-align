import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# import IPython; IPython.embed(); exit()


class Resnet(nn.Module):
    def __init__(self, channels=1, pretrained=True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(512, 6, bias=True)

    def forward(self, x):
        return self.resnet(x)


class Resnet50(nn.Module):
    def __init__(self, channels=1, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(512, 6, bias=True)

    def forward(self, x):
        return self.resnet(x)


class SlidingWindowHourglassNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)  # downsample * 6
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.down = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        self.up = self._make_up_layers(512, [256, 256, 256, 128, 128])
        self.final_layer = nn.Conv2d(  # channels: confidence, x1, y1, x2, y2, x3, y3
            in_channels=128,
            out_channels=7,
            kernel_size=3,
            stride=1,
            padding=1)

    def _make_up_layers(self, in_planes, num_planes):
        layers = []
        for planes in num_planes:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return self.final_layer(x)


class HourglassNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)  # downsample * 6
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.down = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        self.up = self._make_up_layers(512, [256, 256, 256, 128, 128])
        self.final_layer = nn.Conv2d(
            in_channels=128,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1)

    def _make_up_layers(self, in_planes, num_planes):
        layers = []
        for planes in num_planes:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return self.final_layer(x)


if '__main__' == __name__:
    m = Resnet()
    import IPython
    IPython.embed()
    exit()
