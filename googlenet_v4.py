# coding:utf8

import sys
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out, kernel_size=kernel_size, stride=stride, padding=padding),
                                  nn.BatchNorm2d(out),
                                  nn.ReLU(inplace=True))
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # return F.relu(x, inplace=True)
        return x


class GoogLeNetV4Stem(nn.Module):
    def __init__(self):
        super(GoogLeNetV4Stem, self).__init__()
        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool4_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4_2 = BasicConv2d(64, 96, kernel_size=3, stride=2, padding=0)

        self.conv5_1_1 = BasicConv2d(160, 64, kernel_size=1, stride=1, padding=0)
        self.conv5_1_2 = BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv5_1_3 = BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.conv5_1_4 = BasicConv2d(64, 96, kernel_size=(3, 3), stride=1, padding=0)

        self.conv5_2_1 = BasicConv2d(160, 64, kernel_size=1, stride=1, padding=0)
        self.conv5_2_2 = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=0)

        self.pool6_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv6_2 = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        # print x.shape
        x = self.conv2(x)
        # print x.shape
        x = self.conv3(x)
        # print x.shape

        x1 = self.pool4_1(x)
        # print x1.shape
        x2 = self.conv4_2(x)
        # print x2.shape
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv5_1_1(x)
        x1 = self.conv5_1_2(x1)
        x1 = self.conv5_1_3(x1)
        x1 = self.conv5_1_4(x1)
        # print x1.shape
        x2 = self.conv5_2_1(x)
        x2 = self.conv5_2_2(x2)
        # print x2.shape
        x = torch.cat((x1, x2), dim=1)
        # print x.shape
        x1 = self.conv6_2(x)
        x2 = self.pool6_1(x)
        x = torch.cat((x1, x2), dim=1)
        # print x.shape

        return x


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.conv1 = nn.Sequential(BasicConv2d(384, 96, kernel_size=1, stride=1, padding=0))

        self.conv2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))

        self.conv3 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))

        self.conv4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(384, 96, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


# 35*35 -> 17*17
class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.conv1 = BasicConv2d(384, 384, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1, padding=1),
                                   BasicConv2d(192, 224, kernel_size=3, stride=1, padding=0),
                                   BasicConv2d(224, 256, kernel_size=3, stride=2, padding=0))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.pool3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.conv1 = BasicConv2d(1024, 384, kernel_size=1, stride=1, padding=0)

        self.conv2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                   BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv3 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                   BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                   BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                   BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.conv4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(1024, 128, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


# 17*17 -> 8*8
class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.conv1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(192, 192, kernel_size=3, stride=2, padding=0))
        self.conv2 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                   BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                   BasicConv2d(320, 320, kernel_size=3, stride=2, padding=0))
        self.conv3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)



    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
        pass

    pass


class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()
        self.conv1 = BasicConv2d(1536, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = BasicConv2d(1536, 384, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv2_2 = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Sequential(BasicConv2d(1536, 384, kernel_size=1, stride=1, padding=0),
                                   BasicConv2d(384, 448, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                                   BasicConv2d(448, 512, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv3_1 = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3_2 = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   BasicConv2d(1536, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2_1 = self.conv2_1(x2)
        x2_2 = self.conv2_2(x2)
        x3 = self.conv3(x)
        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_2(x3)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2_1, x2_2, x3_1, x3_2, x4], dim=1)
        return x


class GoogLeNetV4(nn.Module):
    def __init__(self):
        super(GoogLeNetV4, self).__init__()
        self.stem = GoogLeNetV4Stem()
        self.inceptionA1 = InceptionA()
        self.inceptionA2 = InceptionA()
        self.inceptionA3 = InceptionA()
        self.inceptionA4 = InceptionA()
        self.reductionA = ReductionA()
        self.inceptionB1 = InceptionB()
        self.inceptionB2 = InceptionB()
        self.inceptionB3 = InceptionB()
        self.inceptionB4 = InceptionB()
        self.inceptionB5 = InceptionB()
        self.inceptionB6 = InceptionB()
        self.inceptionB7 = InceptionB()
        self.reductionB = ReductionB()
        self.inceptionC1 = InceptionC()
        self.inceptionC2 = InceptionC()
        self.inceptionC3 = InceptionC()
        self.global_pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1536, 1000)

    def forward(self, x):
        print x.shape
        x = self.stem(x)
        print x.shape
        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)
        x = self.inceptionA4(x)
        print x.shape
        x = self.reductionA(x)
        print x.shape
        x = self.inceptionB1(x)
        x = self.inceptionB2(x)
        x = self.inceptionB3(x)
        x = self.inceptionB4(x)
        x = self.inceptionB5(x)
        x = self.inceptionB6(x)
        x = self.inceptionB7(x)
        print x.shape
        x = self.reductionB(x)
        print x.shape
        x = self.inceptionC1(x)
        x = self.inceptionC2(x)
        x = self.inceptionC3(x)
        print x.shape
        x = self.global_pool(x)
        print x.shape
        x = x.view(x.size()[0], -1)
        print x.shape
        x = self.dropout(x)
        x = self.fc(x)
        print x.shape
        return x



if __name__ == '__main__':
    input_tensor = torch.randn((1, 3, 299, 299))
    input_var = torch.autograd.Variable(input_tensor)

    model = GoogLeNetV4()
    output = model(input_var)



