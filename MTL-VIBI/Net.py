import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.block1_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.block1_relu1 = nn.ReLU(inplace=True) 
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.block1_relu2 = nn.ReLU(inplace=True) 
        self.block1_pool1 = nn.MaxPool2d(2, 2)
                                       
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.block2_relu1 = nn.ReLU(inplace=True) 
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.block2_relu2 = nn.ReLU(inplace=True) 
        self.block2_pool1 = nn.MaxPool2d(2, 2)
        
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.block3_relu1 = nn.ReLU(inplace=True) 
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block3_relu2 = nn.ReLU(inplace=True) 
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block3_relu3 = nn.ReLU(inplace=True) 
        self.block3_pool1 = nn.MaxPool2d(2, 2)

        # 定义分类分支的第四五个卷积块
        self.c_block4_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.c_block4_relu1 = nn.ReLU(inplace=True) 
        self.c_block4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.c_block4_relu2 = nn.ReLU(inplace=True) 
        self.c_block4_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.c_block4_bn = nn.BatchNorm2d(256)
        self.c_block4_relu3 = nn.ReLU(inplace=True) 
        self.c_block4_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c_block5_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.c_block5_relu1 = nn.ReLU(inplace=True) 
        self.c_block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.c_block5_relu2 = nn.ReLU(inplace=True) 
        self.c_block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.c_block5_bn = nn.BatchNorm2d(512)
        self.c_block5_relu3 = nn.ReLU(inplace=True) 
        self.c_block5_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 定义分类分支的分类器
        self.classifier = torch.nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
#             nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2)
        )
        
        # 定义分割分支的第四五个卷积块
        self.s_block4_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.s_block4_relu1 = nn.ReLU(inplace=True) 
        self.s_block4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.s_block4_relu2 = nn.ReLU(inplace=True) 
        self.s_block4_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.s_block4_bn = nn.BatchNorm2d(256)
        self.s_block4_relu3 = nn.ReLU(inplace=True) 
        self.s_block4_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.s_block5_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.s_block5_relu1 = nn.ReLU(inplace=True) 
        self.s_block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.s_block5_relu2 = nn.ReLU(inplace=True) 
        self.s_block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.s_block5_bn = nn.BatchNorm2d(512)
        self.s_block5_relu3 = nn.ReLU(inplace=True) 
        

        # 定义上采样部分网络
        self.up1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.up1_bn1 = nn.BatchNorm2d(256)
        self.up1_relu1 = nn.ReLU(inplace=True) 
        self.up1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.up1_bn2 = nn.BatchNorm2d(256)
        self.up1_relu2 = nn.ReLU(inplace=True) 
        
        self.up2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.up2_bn1 = nn.BatchNorm2d(256)
        self.up2_relu1 = nn.ReLU(inplace=True) 
        self.up2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(256)
        self.up2_relu2 = nn.ReLU(inplace=True) 
        
        self.up3_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.up3_bn1 = nn.BatchNorm2d(128)
        self.up3_relu1 = nn.ReLU(inplace=True) 
        self.up3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.up3_bn2 = nn.BatchNorm2d(128)
        self.up3_relu2 = nn.ReLU(inplace=True) 
        
        self.up4_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.up4_bn1 = nn.BatchNorm2d(64)
        self.up4_relu1 = nn.ReLU(inplace=True) 
        self.up4_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.up4_bn2 = nn.BatchNorm2d(64)
        self.up4_relu2 = nn.ReLU(inplace=True) 
        
        self.fusion1 = torch.nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = torch.nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 最后是1x1的卷积，用于将通道数化为3
        self.seg_outconv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        
         
    def forward(self, x):

        # 1：进行共享编码过程
        x = self.block1_conv1(x)
        x = self.block1_relu1(x)
        x = self.block1_conv2(x)
        feature_block1 = self.block1_relu2(x)
        x = self.block1_pool1(feature_block1)
        
        x = self.block2_conv1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        feature_block2 = self.block1_relu2(x)
        x = self.block2_pool1(feature_block2)
        
        x = self.block3_conv1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_relu2(x)
        x = self.block3_conv3(x)
        feature_block3 = self.block3_relu3(x)
        x = self.block3_pool1(feature_block3)

        # 分类分支
        x1 = x
        x1 = self.c_block4_conv1(x1)
        x1 = self.c_block4_relu1(x1)
        x1 = self.c_block4_conv2(x1)
        x1 = self.c_block4_relu2(x1)
        x1 = self.c_block4_conv3(x1)
        c_feature_block4 = self.c_block4_relu3(x1)
        x1 = self.c_block4_bn(c_feature_block4)
        x1 = self.c_block4_pool1(x1)
        
        x1 = self.c_block5_conv1(x1)
        x1 = self.c_block5_relu1(x1)
        x1 = self.c_block5_conv2(x1)
        x1 = self.c_block5_relu2(x1)
        x1 = self.c_block5_conv3(x1)
        c_feature_block5 = self.c_block4_relu3(x1)
        x1 = self.c_block5_bn(c_feature_block5)
        x1 = self.c_block5_pool1(x1)
        x1 = x1.view(x1.size(0),-1)
        clas_out = self.classifier(x1)
        clas_out = nn.LogSoftmax(dim=1)(clas_out)
        
        # 分割分支
        # s_block4
        x2 = x
        x2 = self.s_block4_conv1(x2)
        x2 = self.s_block4_relu1(x2)
        x2 = self.s_block4_conv2(x2)
        x2 = self.s_block4_relu2(x2)
        x2 = self.s_block4_conv3(x2)
        s_feature_block4 = self.s_block4_relu3(x2) # torch.Size([16, 256, 28, 28])
        # cross1    
        x2 = torch.matmul(s_feature_block4, c_feature_block4) # torch.Size([16, 256, 28, 28])
        x2 = torch.cat((s_feature_block4, x2), dim=1) # torch.Size([16, 256, 28, 28])
        x2 = self.fusion1(x2)  
        x2 = self.s_block4_pool1(x2)

        # s_block5
        x2 = self.s_block5_conv1(x2)
        x2 = self.s_block5_relu1(x2)
        x2 = self.s_block5_conv2(x2)
        x2 = self.s_block5_relu2(x2)
        x2 = self.s_block5_conv3(x2)
        s_feature_block5 = self.s_block5_relu3(x2)
        # cross2    
        x2 = torch.matmul(s_feature_block5, c_feature_block5) 
        x2 = torch.cat((s_feature_block5, x2), dim=1)
        x2 = self.fusion2(x2)
        
        
        
        # 进行解码过程
        # UP1
        x2 = self.up1_1(x2)
        temp = torch.cat((s_feature_block4, x2), dim=1)
        x2 = self.up1_conv1(temp)
        x2 = self.up1_bn1(x2)
        x2 = self.up1_relu1(x2)
        x2 = self.up1_conv2(x2)
        x2 = self.up1_bn2(x2)
        x2 = self.up1_relu2(x2)
        
        x2 = self.up2_1(x2)
        temp = torch.cat((feature_block3, x2), dim=1)
        x2 = self.up2_conv1(temp)
        x2 = self.up2_bn1(x2)
        x2 = self.up2_relu1(x2)
        x2 = self.up2_conv2(x2)
        x2 = self.up2_bn2(x2)
        x2 = self.up2_relu2(x2)
        
        x2 = self.up3_1(x2)
        temp = torch.cat((feature_block2, x2), dim=1)
        x2 = self.up3_conv1(temp)
        x2 = self.up3_bn1(x2)
        x2 = self.up3_relu1(x2)
        x2 = self.up3_conv2(x2)
        x2 = self.up3_bn2(x2)
        x2 = self.up3_relu2(x2)
        
        x2 = self.up4_1(x2)
        temp = torch.cat((feature_block1, x2), dim=1)
        x2 = self.up4_conv1(temp)
        x2 = self.up4_bn1(x2)
        x2 = self.up4_relu1(x2)
        x2 = self.up4_conv2(x2)
        x2 = self.up4_bn2(x2)
        x2 = self.up4_relu2(x2)
        
        seg_out = self.seg_outconv(x2)
        seg_out = nn.LogSoftmax(dim=1)(seg_out)


        return clas_out, seg_out
