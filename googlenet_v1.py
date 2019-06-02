# coding:utf8
import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()

        # Inception(192, 64, 96, 128, 16, 32, 32)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1, padding=1)
        )

    def forward(self, x):
        # print x.shape
        x1 = self.branch1(x)
        # print x1.shape
        x2 = self.branch2(x)
        # print x1.shape
        x3 = self.branch3(x)
        # print x3.shape
        x4 = self.branch4(x)
        # print x4.shape
        print x1.shape, x2.shape, x3.shape, x4.shape
        x = torch.cat([x1, x2, x3, x4], 1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.inception_3 = nn.Sequential(Inception(192, 64, 96, 128, 16, 32, 32),
                                         Inception(256, 128, 128, 192, 32, 96, 64),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_4 = nn.Sequential(Inception(480, 192, 96, 208, 16, 48, 64),
                                         Inception(512, 160, 112, 224, 24, 64, 64),
                                         Inception(512, 128, 128, 256, 24, 64, 64),
                                         Inception(512, 112, 144, 288, 32, 64, 64),
                                         Inception(528, 256, 160, 320, 32, 128, 128),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_5 = nn.Sequential(Inception(832, 256, 160, 320, 32, 128, 128),
                                         Inception(832, 384, 192, 384, 48, 128, 128))

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, 1000)



    def forward(self, x):
        print x.shape
        x = self.conv1(x)
        # print x.shape
        x = self.conv2(x)
        # print x.shape
        x = self.inception_3(x)
        print x.shape
        x = self.inception_4(x)
        print x.shape
        x = self.inception_5(x)
        print x.shape
        x = self.avg_pool(x)
        print x.shape
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        print x.shape
        x = self.dropout(x)
        x = self.linear(x)
        print x.shape
        return

if __name__ == '__main__':
    input_tensor = torch.randn((1, 3, 224, 224))
    input_var = torch.autograd.Variable(input_tensor)

    model = GoogLeNet()
    out = model(input_var)



