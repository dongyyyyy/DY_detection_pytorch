import torch.nn as nn
import torch
from torchvision.models import vgg19
from torchsummary import summary

class VGG19(nn.Module): # VGG19
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19_model = vgg19(pretrained=True) # pretrained된 vgg19 model
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img): # forward
        return self.feature_extractor(img)

class ResidualBlock_v1(nn.Module): # ResNet BasicBlock
    def __init__(self, in_features,out_features,stride=1):
        super(ResidualBlock_v1, self).__init__()
        self.downsample = stride
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1,bias=False), # 3X3 conv filter = same
            nn.BatchNorm2d(out_features, 0.8), # batch normalization
            nn.PReLU(), # Leakly ReLU
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1,bias=False), # 3X3 conv filter = same
            nn.BatchNorm2d(out_features, 0.8), # batch normalization
        )
        if stride == 2:
            self.downsampling = nn.Conv2d(in_features,out_features,kernel_size=1,stride=2)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        conv_out = self.conv_block(x)
        if(self.downsample == 2):
            x = self.downsampling(x)
        return self.ReLU(x + conv_out)# concat ( shortcut connection )

class ResidualBlock_v2(nn.Module): # ResNet BasicBlock
    def __init__(self, in_features, out_features, stride=1):
        super(ResidualBlock_v2, self).__init__()

        self.downsample = stride

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1, stride=stride, bias=False), # 3X3 conv filter = same
            nn.BatchNorm2d(in_features, 0.8), # batch normalization
            nn.ReLU(), # Leakly ReLU
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False), # 3X3 conv filter = same
            nn.BatchNorm2d(in_features, 0.8), # batch normalization
            nn.ReLU(),  # Leakly ReLU
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False),  # 3X3 conv filter = same
            nn.BatchNorm2d(out_features, 0.8),  # batch normalization
        )

        self.ReLU = nn.ReLU()
        if stride == 2:
            self.downsampling = nn.Conv2d(in_features,out_features,kernel_size=1,stride=2)
        self.none_downsampling = nn.Conv2d(in_features,out_features,kernel_size=1,stride=1)

    def forward(self, x):
        conv_out = self.conv_block(x)
        if(self.downsample == 2):
            x = self.downsampling(x)
        else:
            x = self.none_downsampling(x)
        print(x.shape)

        #print(x.shape)
        #print(conv_out.shape)
        return self.ReLU(x + conv_out)# concat ( shortcut connection )


class ResNet(nn.Module):
    def __init__(self, input_shape, n_residual_blocks,basic_block = 'v1'):
        super(ResNet,self).__init__()

        in_channels = input_shape
        if(basic_block=='v1'):
            output_channels = 512
        else:
            output_channels = 2048
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
        first_res_blocks = []
        for _ in range(n_residual_blocks[0]):
            if(basic_block=='v1'):
                first_res_blocks.append(ResidualBlock_v1(64,64))
            else:
                first_res_blocks.append(ResidualBlock_v2(64, 256))

        second_res_blocks = []
        for i in range(n_residual_blocks[1]):
            if(basic_block=='v1'):
                if(i == 0):
                    second_res_blocks.append(ResidualBlock_v1(64,128,stride=2))
                else:
                    second_res_blocks.append(ResidualBlock_v1(128,128))
            else:
                if (i == 0):
                    second_res_blocks.append(ResidualBlock_v2(128, 512, stride=2))
                else:
                    second_res_blocks.append(ResidualBlock_v2(128, 512))
        thrid_res_blocks = []
        for i in range(n_residual_blocks[2]):
            if(basic_block=='v1'):
                if (i == 0):
                    thrid_res_blocks.append(ResidualBlock_v1(128,256, stride=2))
                else:
                    thrid_res_blocks.append(ResidualBlock_v1(256,256))
            else:
                if (i == 0):
                    thrid_res_blocks.append(ResidualBlock_v2(256, 1024, stride=2))
                else:
                    thrid_res_blocks.append(ResidualBlock_v2(256, 1024))
        firth_res_blocks = []
        for i in range(n_residual_blocks[3]):
            if(basic_block=='v1'):
                if (i == 0):
                    firth_res_blocks.append(ResidualBlock_v1(256,512, stride=2))
                else:
                    firth_res_blocks.append(ResidualBlock_v1(512,512))
            else:
                if (i == 0):
                    firth_res_blocks.append(ResidualBlock_v2(512, 2048, stride=2))
                else:
                    firth_res_blocks.append(ResidualBlock_v2(512, 2048))
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            *first_res_blocks
        )

        self.conv3 = nn.Sequential(
            *second_res_blocks
        )

        self.conv4 = nn.Sequential(
            *thrid_res_blocks
        )

        self.conv5 = nn.Sequential(
            *firth_res_blocks
        )

        self.averagepool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(output_channels*7*7,1000)

    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.averagepool(x)
        print(x.shape)
        x = torch.flatten(x,1)
        print(x.shape)
        x = self.fc(x)

        return x

if __name__=='__main__':
    #resnet18 =ResNet(input_shape=3, n_residual_blocks=[2,2,2,2], basic_block='v1')
    #summary(resnet18.cuda(),input_size=(3,224,224))

    resnet101 = ResNet(input_shape=3, n_residual_blocks=[3,4,23,3], basic_block='v2')
    summary(resnet101.cuda(),input_size=(3,224,224))