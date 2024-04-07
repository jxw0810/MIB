import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, i, in_channels, middle_channels, out_channels):
        super().__init__()

        conv_relu = []
        if i <= 3:
            conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                       kernel_size=(3, 3), padding=1, stride=(1, 1)))
            conv_relu.append(nn.ReLU())
            conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                       kernel_size=(3, 3), padding=1, stride=(1, 1)))
            conv_relu.append(nn.ReLU())
        else:
            conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                       kernel_size=(3, 3), padding=1, stride=(1, 1)))
            conv_relu.append(nn.ReLU())
            conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                       kernel_size=(3, 3), padding=1, stride=(1, 1)))
            conv_relu.append(nn.ReLU())
            conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                       kernel_size=(3, 3), padding=1, stride=(1, 1)))
            conv_relu.append(nn.ReLU())

        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = ConvBlock(i=1, in_channels=1, middle_channels=64, out_channels=64)
        self.left_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(i=2, in_channels=64, middle_channels=128, out_channels=128)
        self.left_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(i=3, in_channels=128, middle_channels=256, out_channels=256)
        self.left_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义分类分支的第四五个卷积块
        self.conv_4 = ConvBlock(i=4, in_channels=256, middle_channels=256, out_channels=256)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = ConvBlock(i=5, in_channels=256, middle_channels=512, out_channels=512)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义分割分支的第四五个卷积块
        self.left_conv_4s = ConvBlock(i=4, in_channels=256, middle_channels=256, out_channels=256)
        self.pool_4s = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5s = ConvBlock(i=5, in_channels=256, middle_channels=512, out_channels=512)


        # 定义右半部分网络
        self.deconv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.right_conv_1 = ConvBlock(i=1, in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.right_conv_2 = ConvBlock(i=1, in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2 ,output_padding=1)
        self.right_conv_3 = ConvBlock(i=1, in_channels=256, middle_channels=128, out_channels=128)

        self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.right_conv_4 = ConvBlock(i=1, in_channels=128, middle_channels=64, out_channels=64)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

        self.classifier = torch.nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
#             nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2)
        )


    def forward(self, x):

        # 1：进行共享编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.left_pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.left_pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.left_pool_3(feature_3)

        # 分类分支
        feature_4c = self.conv_4(feature_3_pool)
        feature_4c_pool = self.pool_4(feature_4c)

        feature_5c = self.conv_5(feature_4c_pool)
        feature_5c_pool = self.pool_5(feature_5c)

        feature_5c_pool2 = feature_5c_pool.view(feature_5c_pool.size(0), -1)
        clas_out = self.classifier(feature_5c_pool2)

        # 分割分支
        feature_4s = self.left_conv_4s(feature_3_pool)
        feature_4s_pool = self.pool_4s(feature_4s)

        feature_5s = self.left_conv_5s(feature_4s_pool)

        # 进行解码过程
        de_feature_1 = self.deconv_1(feature_5s)  # (8, 512, 28, 28)
        # 特征拼接
        temp = torch.cat((feature_4s, de_feature_1), dim=1)  # (256, )
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        seg_out = self.right_conv_5(de_feature_4_conv)
#         seg_out = nn.Softmax(seg_out)
        # s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same', name='segmentation_output')(o)


        return clas_out, seg_out
