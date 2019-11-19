import torch.nn as nn
import torch

class BasicBlock(nn.Module):  # BasicBlock ( ResNet-18 & ResNet-34 )
    def __init__(self, in_features, out_features, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = stride
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_features, 0.8),  # batch normalization
            nn.ReLU(),  #  ReLU
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features, 0.8),  # batch normalization
        )
        if stride == 2:
            self.downsampling = nn.Conv2d(in_features, out_features, kernel_size=1, stride=2)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        block_out = self.conv_block(x)
        if (self.downsample == 2):
            x = self.downsampling(x)
        out = self.ReLU(x + block_out)  # plus ( shortcut connection )
        return out


class BottleNeck(nn.Module):  # Bottleneck ( ResNet-50 & ResNet-101 & ResNet-152 )
    def __init__(self, in_features, out_features, stride=1):
        super(BottleNeck, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if stride == 2:
            self.downsample = True  # DownSampling을 해야하는지에 대한 정보
        else:
            self.downsample = False
        self.block_features = self.out_features // 4
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
        if self.downsample:  #stride가 2인 경우에 input의 값을 다운샘플링 해주어야 함.
            self.downsampling = nn.Conv2d(self.in_features, out_features, kernel_size=1,
                                          stride=2)  # 따라서 1X1 conv의 stride=2를 통하여 down sampling을 진행
        else:
            self.downsampling = nn.Conv2d(self.in_features, out_features, kernel_size=1,
                                          stride=1)  # 그 외의 경우에는 down sampling을 하지 않고 filter만 확장

    def forward(self, input):
        #params = list(self.conv_block.parameters())
        #print("conv1 weight: ",params[0].size())
        block_out = self.conv_block(input)
        x = self.downsampling(input)
        out = self.ReLU(x + block_out)  # plus ( shortcut connection )
        return out