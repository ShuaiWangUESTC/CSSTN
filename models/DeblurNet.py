#!/usr/bin/python
# -*- coding: utf-8 -*-
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
# Modified by Qingqing Chen <qingqingchen618@gmail.com>

from models.submodules import *
from models.FAC.kernelconv2d import KernelConv2D
from torch import nn

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        ks = 3
        ks1 = 5
        ks_2d = 3  # adaptive filter size
        ch1 = 32
        ch2 = 64
        ch3 = 128

        ## feature extraction module
        self.conv1_1 = conv(1, ch1, kernel_size=ks1, stride=1)
        self.conv1_2 = resnet_block(ch1, ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, ch1, kernel_size=ks)
        self.conv1_4 = resnet_block(ch1, ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks1, stride=2)
        self.conv2_2 = resnet_block(ch2, ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, ch2, kernel_size=ks)
        self.conv2_4 = resnet_block(ch2, ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks1, stride=2)
        self.conv3_2 = resnet_block(ch3, ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, ch3, kernel_size=ks)
        self.conv3_4 = resnet_block(ch3, ch3, kernel_size=ks)

        self.kconv_warp = KernelConv2D.KernelConv2D(kernel_size=ks_2d)     # CSFAC
        self.kconv_deblur = KernelConv2D.KernelConv2D(kernel_size=ks_2d)   # CSFAC

        ## reconstruction module
        self.upconv3_1 = resnet_block(2*ch3, ch3, kernel_size=ks)
        self.upconv3_2 = resnet_block(ch3, ch3, kernel_size=ks)
        self.upconv3_3 = resnet_block(ch3, ch3, kernel_size=ks)
        self.upconv3_u = upconv(ch3, ch2)

        self.upconv2_1 = resnet_block(ch2, ch2, kernel_size=ks)
        self.upconv2_2 = resnet_block(ch2, ch2, kernel_size=ks)
        self.upconv2_3 = resnet_block(ch2, ch2, kernel_size=ks)
        self.upconv2_u = upconv(ch2, ch1)

        self.upconv1_1 = resnet_block(ch1, ch1, kernel_size=ks)
        self.upconv1_2 = resnet_block(ch1, ch1, kernel_size=ks)
        self.upconv1_3 = resnet_block(ch1, ch1, kernel_size=ks)
        self.upconv1_u = conv(ch1, 1, kernel_size=ks1, stride=1)

        ## kernel prediction module

        # encoder
        self.kconv1_1 = conv(3, ch1, kernel_size=ks1, stride=1)
        self.kconv1_2 = resnet_block(ch1, ch1, kernel_size=ks)
        self.kconv1_3 = resnet_block(ch1, ch1, kernel_size=ks)
        self.kconv1_4 = resnet_block(ch1, ch1, kernel_size=ks)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks1, stride=2)
        self.kconv2_2 = resnet_block(ch2, ch2, kernel_size=ks)
        self.kconv2_3 = resnet_block(ch2, ch2, kernel_size=ks)
        self.kconv2_4 = resnet_block(ch2, ch2, kernel_size=ks)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks1, stride=2)
        self.kconv3_2 = resnet_block(ch3, ch3, kernel_size=ks)
        self.kconv3_3 = resnet_block(ch3, ch3, kernel_size=ks)
        self.kconv3_4 = resnet_block(ch3, ch3, kernel_size=ks)

        # solve Fsvi
        self.fac_warp = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, ch3, kernel_size=ks),
            conv(ch3, 1 * ks_2d ** 2, kernel_size=ks))

        self.kconv4 = conv(1 * ks_2d ** 2, ch3, kernel_size=1)

        # solve Fsvc
        self.fac_deblur = nn.Sequential(
            conv(2 * ch3, ch3, kernel_size=ks),
            resnet_block(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, ch3, kernel_size=ks),
            conv(ch3, 1 * ks_2d ** 2, kernel_size=ks))

        # previous time step feature output
        self.fea = conv(2 * ch3, ch3, kernel_size=ks, stride=1)

    def forward(self, img_blur, last_img_blur, output_last_img, output_last_fea):
        ## kernel prediction
        merge = torch.cat([img_blur, last_img_blur, output_last_img], 1)

        # encoder
        kconv1 = self.kconv1_4(self.kconv1_3(self.kconv1_2(self.kconv1_1(merge))))
        kconv2 = self.kconv2_4(self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1))))
        kconv3 = self.kconv3_4(self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2))))

        # Fsvi
        channel = 128
        kernel_warp = self.fac_warp(kconv3)                              # channel -> k^2
        kernel_warp_cp = kernel_warp.repeat(1, channel, 1, 1)            # channel -> 128*k^2

        # Fsvc
        kconv4 = self.kconv4(kernel_warp)
        kernel_deblur = self.fac_deblur(torch.cat([kconv3,kconv4],1))    # channel -> k^2
        kernel_deblur_cp= kernel_deblur.repeat(1, channel, 1, 1)         # channel -> 128*k^2

        ## feature extraction
        conv1_d = self.conv1_1(img_blur)
        conv1_d = self.conv1_4(self.conv1_3(self.conv1_2(conv1_d)))

        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_4(self.conv2_3(self.conv2_2(conv2_d)))

        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_4(self.conv3_3(self.conv3_2(conv3_d)))     # feature extracted from current image

        conv3_d_k = self.kconv_deblur(conv3_d, kernel_deblur_cp)        # CSFAC

        # encoder last_clear
        if output_last_fea is None:
            output_last_fea = torch.cat([conv3_d, conv3_d],1)

        output_last_fea = self.fea(output_last_fea)         # previous time step feature output

        conv_a_k = self.kconv_warp(output_last_fea, kernel_warp_cp)   # CSFAC

        conv3 = torch.cat([conv3_d_k, conv_a_k],1)

        # reconstruction
        upconv3 = self.upconv3_u(self.upconv3_3(self.upconv3_2(self.upconv3_1(conv3))))
        upconv2 = self.upconv2_u(self.upconv2_3(self.upconv2_2(self.upconv2_1(upconv3 + conv2_d))))
        upconv1 = self.upconv1_u(self.upconv1_3(self.upconv1_2(self.upconv1_1(upconv2 + conv1_d))))

        output_img = upconv1 + img_blur
        output_fea = conv3   # CSFAC  result

        return output_img, output_fea
