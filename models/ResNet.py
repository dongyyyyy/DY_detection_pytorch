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
        # params = list(self.conv_block.parameters())
        # print("conv1 weight: ",params[0].size())
        block_out = self.conv_block(input)
        x = self.downsampling(input)
        out = self.ReLU(x + block_out)  # plus ( shortcut connection )
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, n_residual_blocks, basic_block=BasicBlock, classes=1000):     #input_shape = 입력 이미지 (학습) / n_residual_blocks = ResNet block 개수 (리스트)
        super(ResNet, self).__init__()                                                      #basic_block = [v1:"BasicBlock",v2:"Bottleneck"] / classes = class개수

        in_channels = input_shape

        self.blocks = basic_block
        if self.blocks == basic_block: #Basic_block의 종류에 따라서 Feature Extract output 크기가 다름
            output_channels = 512
        else:
            output_channels = 2048
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        if self.blocks == basic_block:
            self.resnet_blocks1 = self._make_layer(in_filters=64, out_filters=64, blocks=n_residual_blocks[0])
        else:
            self.resnet_blocks1 = self._make_layer(in_filters=64, out_filters=256, blocks=n_residual_blocks[0])

        if self.blocks == basic_block:
            self.resnet_blocks2 = self._make_layer(in_filters=64, out_filters=128, blocks=n_residual_blocks[1], stride=2)
        else:
            self.resnet_blocks2 = self._make_layer(in_filters=128, out_filters=512, blocks=n_residual_blocks[1], stride=2)

        if self.blocks == basic_block:
            self.resnet_blocks3 = self._make_layer(in_filters=128, out_filters=256, blocks=n_residual_blocks[2], stride=2)
        else:
            self.resnet_blocks3 = self._make_layer(in_filters=256, out_filters=1024, blocks=n_residual_blocks[2], stride=2)

        if self.blocks == basic_block:
            self.resnet_blocks4 = self._make_layer(in_filters=256, out_filters=512, blocks=n_residual_blocks[3], stride=2)
        else:
            self.resnet_blocks4 = self._make_layer(in_filters=512, out_filters=2048, blocks=n_residual_blocks[3], stride=2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.averagepool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # output_size에 맞게 자동으로 kernel크기 설정
        self.fc = nn.Linear(output_channels, classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.max_pool(x)
        x = self.resnet_blocks1(x)
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.averagepool(x)  # average pool output = ( batch , channel, 1 , 1 )
        x = torch.flatten(x, 1) # (batch, channel)
        out = self.fc(x) # (batch , classes )

        return out

    def _make_layer(self, in_filters, out_filters, blocks,stride=1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(self.blocks(in_features=in_filters,out_features=out_filters,stride=stride))
            else:
                layers.append(self.blocks(in_features=out_filters, out_features=out_filters))
        return nn.Sequential(*layers)

def get_resnet18():
    return ResNet(input_shape=3, n_residual_blocks=[2, 2, 2, 2], basic_block=BasicBlock)


def get_resnet34():
    return ResNet(input_shape=3, n_residual_blocks=[3, 4, 6, 3], basic_block=BasicBlock)


def get_resnet50():
    return ResNet(input_shape=3, n_residual_blocks=[3, 4, 6, 3], basic_block=BottleNeck)


def get_resnet101():
    return ResNet(input_shape=3, n_residual_blocks=[3, 4, 23, 3], basic_block=BottleNeck)


def get_resnet152():
    return ResNet(input_shape=3, n_residual_blocks=[3, 8, 36, 3], basic_block=BottleNeck)



'''
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
        # params = list(self.conv_block.parameters())
        # print("conv1 weight: ",params[0].size())
        block_out = self.conv_block(input)
        x = self.downsampling(input)
        out = self.ReLU(x + block_out)  # plus ( shortcut connection )
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, n_residual_blocks, basic_block='v1', classes=1000):     #input_shape = 입력 이미지 (학습) / n_residual_blocks = ResNet block 개수 (리스트)
        super(ResNet, self).__init__()                                                      #basic_block = [v1:"BasicBlock",v2:"Bottleneck"] / classes = class개수

        in_channels = input_shape
        if basic_block == 'v1': #Basic_block의 종류에 따라서 Feature Extract output 크기가 다름
            output_channels = 512
        else:
            output_channels = 2048
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        first_res_blocks = []
        for i in range(n_residual_blocks[0]):
            if basic_block == 'v1':
                first_res_blocks.append(BasicBlock(64, 64))
            else:
                if (i == 0):
                    first_res_blocks.append(BottleNeck(in_features=64, out_features=256))
                else:
                    first_res_blocks.append(BottleNeck(in_features=256, out_features=256))

        second_res_blocks = []
        for i in range(n_residual_blocks[1]):
            if basic_block == 'v1':
                if i == 0:
                    second_res_blocks.append(BasicBlock(64, 128, stride=2))
                else:
                    second_res_blocks.append(BasicBlock(128, 128))
            else:
                if i == 0:
                    second_res_blocks.append(BottleNeck(256, 512, stride=2))
                else:
                    second_res_blocks.append(BottleNeck(512, 512))
        thrid_res_blocks = []
        for i in range(n_residual_blocks[2]):
            if basic_block == 'v1':
                if i == 0:
                    thrid_res_blocks.append(BasicBlock(128, 256, stride=2))
                else:
                    thrid_res_blocks.append(BasicBlock(256, 256))
            else:
                if i == 0:
                    thrid_res_blocks.append(BottleNeck(512, 1024, stride=2))
                else:
                    thrid_res_blocks.append(BottleNeck(1024, 1024))
        firth_res_blocks = []
        for i in range(n_residual_blocks[3]):
            if basic_block == 'v1':
                if i == 0:
                    firth_res_blocks.append(BasicBlock(256, 512, stride=2))
                else:
                    firth_res_blocks.append(BasicBlock(512, 512))
            else:
                if i == 0:
                    firth_res_blocks.append(BottleNeck(1024, 2048, stride=2))
                else:
                    firth_res_blocks.append(BottleNeck(2048, 2048))
        self.conv2 = nn.Sequential(  # layer name  2번 layer의 경우에 down sampling은 3X3/2 Maxpooling을 통하여 처리함.
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *first_res_blocks
        )

        self.conv3 = nn.Sequential(  # layer name
            *second_res_blocks
        )

        self.conv4 = nn.Sequential(  # layer name
            *thrid_res_blocks
        )

        self.conv5 = nn.Sequential(  # layer name
            *firth_res_blocks
        )

        self.averagepool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # output_size에 맞게 자동으로 kernel크기 설정
        self.fc = nn.Linear(output_channels, classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.averagepool(x)  # average pool output = ( batch , channel, 1 , 1 )
        print(x.shape)
        x = torch.flatten(x, 1) # (batch, channel)
        print(x.shape)
        out = self.fc(x) # (batch , classes )

        return out


def get_resnet18():
    return ResNet(input_shape=3, n_residual_blocks=[2, 2, 2, 2], basic_block='v1')


def get_resnet34():
    return ResNet(input_shape=3, n_residual_blocks=[3, 4, 6, 3], basic_block='v1')


def get_resnet50():
    return ResNet(input_shape=3, n_residual_blocks=[3, 4, 6, 3], basic_block='v2')


def get_resnet101():
    return ResNet(input_shape=3, n_residual_blocks=[3, 4, 23, 3], basic_block='v2')


def get_resnet152():
    return ResNet(input_shape=3, n_residual_blocks=[3, 8, 36, 3], basic_block='v2')
'''