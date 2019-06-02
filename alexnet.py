#coding:utf8

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_class=4096):
        super(AlexNet, self).__init__()
        self.num_class = num_class
        self.create_model()

    def create_model(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, (11, 11), stride=4, padding=2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((3, 3), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, (5, 5), stride=1, padding=2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((3, 3), stride=2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, (3, 3), stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, (3, 3), stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, (3, 3), stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((3, 3), stride=2))

        self.fc6 = nn.Sequential(nn.Linear(6*6*256, 4096),
                                 nn.Dropout(p=0.6))
        self.fc7 = nn.Sequential(nn.Linear(4096, self.num_class))

    def forward(self, x):
        batch_size = x.size()[0]
        print x.shape
        x = self.conv1(x)
        print x.shape
        x = self.conv2(x)
        print x.shape
        x = self.conv3(x)
        print x.shape
        x = self.conv4(x)
        print x.shape
        x = self.conv5(x)
        print x.shape
        x = x.view(batch_size, -1)
        print x.shape
        x = self.fc6(x)
        print x.shape
        x = self.fc7(x)
        print x.shape
        return x


if __name__ == '__main__':
    input_tensor =  torch.randn((1, 3, 224, 224))
    input_var = torch.autograd.Variable(input_tensor)

    model = AlexNet(num_class=10)
    output = model(input_var)











