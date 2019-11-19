import torch.nn as nn
import torch

class Residual(nn.Module):  # Bottleneck ( ResNet-50 & ResNet-101 & ResNet-152 )
    def __init__(self, in_features, out_features, stride=1):
        super(Residual, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if stride == 2:
            self.downsample = True  # DownSampling을 해야하는지에 대한 정보
        else:
            self.downsample = False
        self.block_features = self.out_features // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=self.block_features, kernel_size=1, bias=False),
            # 3X3 conv filter = same
            nn.BatchNorm2d(self.block_features, 0.8),  # batch normalization
            nn.ReLU(),  # ReLU
            nn.Conv2d(in_channels=self.block_features, out_channels=self.block_features, kernel_size=3, stride=stride,
                      padding=1, bias=False),  # 3X3 conv 만약 해당 블록의 처음일 경우에는 down sample을 해야 하기 때문에 stride = 2임.
            nn.BatchNorm2d(self.block_features, 0.8),  # batch normalization
            nn.ReLU(),  # ReLU
            nn.Conv2d(in_channels=self.block_features, out_channels=self.out_features, kernel_size=1, stride=1,
                      bias=False),  # 3X3 conv filter = same
            nn.BatchNorm2d(self.out_features, 0.8),  # batch normalization
        )

        self.ReLU = nn.ReLU()
        self.downsampling = nn.Conv2d(self.in_features, out_features, kernel_size=1,
                                          stride=1)  # 그 외의 경우에는 down sampling을 하지 않고 filter만 확장

    def forward(self, input):
        #params = list(self.conv_block.parameters())
        #print("conv1 weight: ",params[0].size())
        block_out = self.conv_block(input)
        x = self.downsampling(input)
        out = self.ReLU(x + block_out)  # plus ( shortcut connection )
        return out


def conv3X3(in_filters,out_filters,stride=1,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=3,stride=stride,padding=padding),
            nn.BatchNorm2d(num_features=out_filters, eps=0.8),
            nn.ReLU()
    )

def conv1X1(in_filters,out_filters,stride=1,padding=0):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=1,stride=stride,padding=padding),
            nn.BatchNorm2d(num_features=out_filters, eps=0.8),
            nn.ReLU()
    )


class Hourglass(nn.Module):
    def __init__(self,input_shape=3):
        super(Hourglass,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(num_features=64, eps=0.8),
            nn.ReLU()
        )
        self.residual1 = Residual(in_features=64,out_features=256)
        self.residual2 = Residual(in_features=256, out_features=256)
        self.Upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.Max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,input):
        x = self.conv1(input)
        x = self.residual1(x)
        x = self.Max_pool(x)
        x = self.residual2(x)

        return x


