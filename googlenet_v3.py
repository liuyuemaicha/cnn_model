# coding:utf8

import sys
import torch
import torch.nn as nn

import torchvision.models


class InceptionA(nn.Module):
    def __init__(self, in_planes, b1_1, b2t_1, b2t_2, b3t_1, b3t_2, b3t_3, b4_1):
        super(InceptionA, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_planes, 48, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(48),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_planes, b4_1, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(b4_1),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        print x.shape
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        print x1.shape, x2.shape, x3.shape, x4.shape

        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_planes, b1_1, b3t_1, b3t_2, b3t_3):
        super(InceptionB, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_planes, 384, kernel_size=3, stride=2, padding=0),
                                     nn.BatchNorm2d(384),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

    def forward(self, x):
        print x.shape
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        print x1.shape, x2.shape, x3.shape

        return torch.cat([x1, x2, x3], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_planes, b1_1, b2t_1, b2t_2, b2t_3, b3t_1, b3t_2, b3t_3, b3t_4, b3t_5, b4_1):
        super(InceptionC, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_planes, 192, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_planes, b2t_1, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(b2t_1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(b2t_1, b2t_1, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                     nn.BatchNorm2d(b2t_1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(b2t_1, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True)
                                     )

        self.branch3 = nn.Sequential(nn.Conv2d(in_planes, b3t_1, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(b3t_1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(b3t_1, b3t_1, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                     nn.BatchNorm2d(b3t_1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(b3t_1, b3t_1, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                     nn.BatchNorm2d(b3t_1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(b3t_1, b3t_1, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                     nn.BatchNorm2d(b3t_1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(b3t_1, 192, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True)
                                     )

        self.branch4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_planes, b4_1, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(b4_1),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        print x.shape
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        print x1.shape, x2.shape, x3.shape, x4.shape

        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionD(nn.Module):
    def __init__(self, in_planes, b1_1, b1_2, b3t_1, b3t_2, b3t_3, b3t_4):
        super(InceptionD, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_planes, 192, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(192, 320, kernel_size=3, stride=2, padding=0),
                                     nn.BatchNorm2d(320),
                                     nn.ReLU(inplace=True)
                                     )

        self.branch2 = nn.Sequential(nn.Conv2d(in_planes, 192, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(192, 192, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

    def forward(self, x):
        print x.shape
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        print x1.shape, x2.shape, x3.shape

        return torch.cat([x1, x2, x3], dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_planes):
        super(InceptionE, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_planes, 320, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(320),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_planes, 384, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(384),
                                     nn.ReLU(inplace=True))
        self.branch2_1 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                       nn.BatchNorm2d(384),
                                       nn.ReLU(inplace=True))
        self.branch2_2 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                                       nn.BatchNorm2d(384),
                                       nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.Conv2d(in_planes, 448, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(448),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(448, 384, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(384),
                                     nn.ReLU(inplace=True))
        self.branch3_1 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                       nn.BatchNorm2d(384),
                                       nn.ReLU(inplace=True))
        self.branch3_2 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                                       nn.BatchNorm2d(384),
                                       nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_planes, 192, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        print x.shape
        x1 = self.branch1(x)

        x2 = self.branch2(x)
        x2_1 = self.branch2_1(x2)
        x2_2 = self.branch2_2(x2)

        x3 = self.branch3(x)
        x3_1 = self.branch3_1(x3)
        x3_2 = self.branch3_2(x3)

        x4 = self.branch4(x)
        print x1.shape, x2.shape, x3.shape, x2_1.shape, x2_2.shape, x3_1.shape, x3_2.shape, x4.shape

        return torch.cat([x1, x2_1, x2_2, x3_1, x3_2, x4], dim=1)


class GoogLeNetV3(nn.Module):
    def __init__(self):
        super(GoogLeNetV3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.conv4 = nn.Sequential(nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(inplace=True))
        # self.conv6 = nn.Sequential(nn.Conv2d(192, 288, kernel_size=3, stride=1, padding=0),
        #                            nn.BatchNorm2d(288),
        #                            nn.ReLU(inplace=True))
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inceptionA1 = InceptionA(192, 64,        48, 64,        64, 96, 96,         32)
        self.inceptionA2 = InceptionA(256, 64,        48, 64,        64, 96, 96,         64)
        self.inceptionA3 = InceptionA(288, 64,        48, 64,        64, 96, 96,         64)

        self.inceptionB = InceptionB(288, 384,        64, 96, 96)

        self.inceptionC1 = InceptionC(768, 192,       128, 128, 192,        128, 128, 128, 128, 192,       192)
        self.inceptionC2 = InceptionC(768, 192,       160, 160, 192,        160, 160, 160, 160, 192,       192)
        self.inceptionC3 = InceptionC(768, 192,       160, 160, 192,        160, 160, 160, 160, 192,       192)
        self.inceptionC4 = InceptionC(768, 192,       192, 192, 192,        192, 192, 192, 192, 192,       192)

        self.inceptionD = InceptionD(768, 192, 320,        192, 192, 192, 192)
        self.inceptionE1 = InceptionE(1280)
        self.inceptionE2 = InceptionE(2048)

        self.gloab_pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.fc = nn.Linear(2048, 1000)
    def forward(self, x):
        x = self.conv1(x)
        print x.shape
        x = self.conv2(x)
        print x.shape
        x = self.conv3(x)
        print x.shape
        x = self.pool3(x)
        print x.shape
        x = self.conv4(x)
        print x.shape
        x = self.conv5(x)
        print x.shape
        x = self.pool6(x)
        print x.shape

        x = self.inceptionA1(x)
        print x.shape
        x = self.inceptionA2(x)
        print x.shape
        x = self.inceptionA3(x)
        print x.shape

        x = self.inceptionB(x)
        print x.shape
        x = self.inceptionC1(x)
        print x.shape
        x = self.inceptionC2(x)
        print x.shape
        x = self.inceptionC3(x)
        print x.shape
        x = self.inceptionC4(x)
        print x.shape

        x = self.inceptionD(x)
        print x.shape
        print 'inceptionD'
        x = self.inceptionE1(x)
        print x.shape
        print 'inceptionE'
        x = self.inceptionE2(x)
        print x.shape
        x = self.gloab_pool(x)
        print x.shape
        x = x.view(x.size()[0], -1)
        print x.shape
        x = self.fc(x)
        print x.shape

        return x


if __name__ == '__main__':
    input_tensor = torch.randn((1, 3, 299, 299))
    input_var = torch.autograd.Variable(input_tensor)
    model = GoogLeNetV3()

    output = model(input_var)
