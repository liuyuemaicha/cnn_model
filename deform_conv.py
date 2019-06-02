# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time



class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=2, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size

        print "x.size: {}".format(x.size())
        print "offset.size: {}".format(offset.size())
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = torch.autograd.Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]),
                                                requires_grad=False).type_as(x).long()
        # print offsets_index
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # print offsets_index.size()
        # print offsets_index.data.numpy()[0]
        # print offset[0]
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # print offset[0]
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)
        print "x zero_padding: {}".format(x.size())
        # print x.data.numpy()[0][0]

        # (b, 2N, h, w)
        print "offset: {}".format(offset.shape)

        p = self._get_p(offset, dtype)
        print "_get_p: {}".format(p.size())

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        print "p permute: {}".format(p.size())
        q_lt = torch.autograd.Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1
        print N

        # print q_lt[..., :N][1]
        # print torch.clamp(q_lt[..., :N], 0, x.size(2)-1)[1]

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        print q_lt.shape, q_rb.shape, q_lb.shape, q_rt.shape
        # (b, h, w, N)

        print p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding)

        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        print mask.shape
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        print x_q_lt.shape, x_q_rb.shape, x_q_lb.shape, x_q_rt.shape
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        print x_offset.shape
        x_offset = self._reshape_x_offset(x_offset, ks)
        print "x_offset: {}".format(x_offset.shape)
        out = self.conv_kernel(x_offset)
        print out.shape
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        print p_n_x.flatten()
        print p_n_y.flatten()
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        print p_n
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = torch.autograd.Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        print p_0_x
        print p_0_y
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        print p_0_x.shape
        print p_0_y.shape

        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        print p_0.shape

        p_0 = torch.autograd.Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        print "p_n: {}".format(p_n.shape)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # print p_0.shape
        # print p_n
        # print p_0
        p = p_0 + p_n
        print p

        # print p[0][0]
        # print offset[0][0]
        # print (p + offset)[0][0]
        return p + offset

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DeformableConv(nn.Module):
    def __init__(self):
        super(DeformableConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        print 'bn1_x: {}'.format(x.data.shape)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        print 'bn2_x: {}'.format(x.data.shape)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        print 'bn3_x: {}'.format(x.data.shape)

        # deformable convolution
        offsets = self.offsets(x)
        print 'offsets: {}'.format(offsets.data.shape)

        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)
        print 'bn4_x: {}'.format(x.data.shape)

        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        print 'avg_pool2d: {}'.format(x.data.shape)

        x = self.classifier(x)
        print 'classifier: {}'.format(x.data.shape)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    img_data = torch.randn((2, 1, 7, 7))
    data_var = torch.autograd.Variable(img_data)

    model = DeformableConv()
    output = model(data_var)

    print output.data.shape

