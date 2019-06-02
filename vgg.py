# coding:utf8

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_name, cfg, num_classes=10, bn=False):
        super(VGG, self).__init__()
        self.vgg_base = self.make_layer(cfg, bn)

        if vgg_name == 'vgg16_C':
            self.fc1 = nn.Sequential(nn.Linear(512 * 8 * 8, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout())
        else:
            self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout())
        self.fc3 = nn.Linear(4096, num_classes)

    def make_layer(self, cfg, bn=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d((2, 2), stride=2)]
            else:
                out_channels, s = v.strip().split('_')
                out_channels, s = int(out_channels), int(s)

                if bn:
                    layers += [nn.Conv2d(in_channels, out_channels, (s, s), padding=1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, out_channels, (s, s), padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.vgg_base(x)
        print x.shape
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


cfg = {
    'vgg11_A': ['64_3', 'M',
                '128_3', 'M',
                '256_3', '256_3', 'M',
                '512_3', '512_3', 'M',
                '512_3', '512_3', 'M'],
    'vgg13_B': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', 'M',
                '512_3', '512_3', 'M',
                '512_3', '512_3', 'M'],
    'vgg16_C': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', '256_1', 'M',
                '512_3', '512_3', '512_1', 'M',
                '512_3', '512_3', '512_1', 'M'],
    'vgg16_D': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', '256_3', 'M',
                '512_3', '512_3', '512_3', 'M',
                '512_3', '512_3', '512_3', 'M'],
    'vgg19_E': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', '256_3', '256_3', 'M',
                '512_3', '512_3', '512_3', '512_3', 'M',
                '512_3', '512_3', '512_3', '512_3', 'M'],
}

if __name__ == '__main__':
    input_tensor = torch.randn((1, 3, 224, 224))
    input_var = torch.autograd.Variable(input_tensor)

    vgg_name = 'vgg19_E'
    model = VGG(vgg_name, cfg[vgg_name], num_classes=10)
    output = model(input_var)
    print output.shape
