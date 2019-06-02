# coding:utf8

import torch
import torch.nn as nn
# import torch.functional as F
import math
from resnet import BasicBlock, Bottleneck


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        # (batch_size, channel, w, h)
        # print 'embed gaussian x shape: {}'.format(x.shape)
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # (batch_size, channel, 1wh)
        # print 'embed gaussian g.view shape: {}'.format(g_x.shape)
        g_x = g_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # (batch_size, channel, 1wh)
        # print 'embed gaussian phi_x.view shape: {}'.format(phi_x.shape)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # (batch_size, channel, 2w*h)
        # print 'embed gaussian theta.view shape: {}'.format(theta_x.shape)
        theta_x = theta_x.permute(0, 2, 1)

        # (batch_size, 2wh, channel) * (batch_size, channel, 1wh) = (batch, 2wh, 1wh)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = self.softmax(f)
        # print 'embed gaussain matmul shape {}'.format(f_div_C.shape)

        # (batch, 2wh, 1wh)*(batch_size, 1wh, channel) = (batch, 2wh, channel)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class ResNet(nn.Module):

    def __init__(self, resnet_block, nonlocal_block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(resnet_block, nonlocal_block, 64, layers[0])
        self.layer2 = self._make_layer(resnet_block, nonlocal_block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(resnet_block, nonlocal_block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(resnet_block, nonlocal_block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(14)
        self.fc = nn.Linear(512 * resnet_block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, resnet_block, nonlocal_block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * resnet_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * resnet_block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * resnet_block.expansion),
            )

        layers = []
        layers.append(resnet_block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * resnet_block.expansion
        for i in range(1, blocks):
            if i != blocks-1:
                layers.append(resnet_block(self.inplanes, planes))
            else:
                layers.append(nonlocal_block(self.inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # [1, 3, 224, 224]
        # print "input x: {}".format(x.shape)
        x = self.conv1(x)
        # [1, 64, 112, 112]
        # print "conv1 x: {}".format(x.shape)
        x = self.bn1(x)
        # [1, 64, 112, 112]
        x = self.relu(x)
        # # [1, 64, 112, 112]
        #x = self.maxpool(x)

        # [1, 64, 112, 112]
        x = self.layer1(x)
        # [1, 64, 112, 112]
        x = self.layer2(x)
        # [1, 128, 56, 56]
        x = self.layer3(x)
        # [1, 256, 28, 28]
        x = self.layer4(x)
        # [1, 512, 14, 14]
        x = self.avgpool(x)
        # [1, 512, 1, 1]
        x = x.view(x.size(0), -1)
        # [1, 512]
        x = self.fc(x)
        # [1, 365]
        return x


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, NONLocalBlock2D, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, NONLocalBlock2D, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    input_tensor = torch.randn((1, 3, 229 ,229))
    input_var = torch.autograd.Variable(input_tensor)
    model = resnet34()
    output = model(input_var)
    print output.shape
