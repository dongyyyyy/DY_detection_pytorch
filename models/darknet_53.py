import torch.nn as nn
import torch

def conv3X3(in_filters,out_filters,stride=1,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=3,stride=stride,padding=padding),
            nn.BatchNorm2d(num_features=out_filters, eps=0.8),
            nn.LeakyReLU()
    )

def conv1X1(in_filters,out_filters,stride=1,padding=0):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=1,stride=stride,padding=padding),
            nn.BatchNorm2d(num_features=out_filters, eps=0.8),
            nn.LeakyReLU()
    )

class darknet_block(nn.Module):
    def __init__(self, filters):
        super(darknet_block, self).__init__()
        self.block_filters = filters//2
        self.conv_block = nn.Sequential(
            conv1X1(in_filters=filters,out_filters=self.block_filters),
            conv3X3(in_filters=self.block_filters,out_filters=filters)
        )

    def forward(self, input):
        block_out = self.conv_block(input)
        out = input + block_out  # plus ( shortcut connection ) activation = Linear [ y = x ]
        return out

class darknet_53(nn.Module):
    def __init__(self,channel=3,n_darknet_blocks=[1,2,8,8,4]):
        super(darknet_53, self).__init__()
        self.channel = channel
        #self.classes = classes

        self.conv1 = nn.Sequential(
            conv3X3(in_filters=self.channel, out_filters=32),
            conv3X3(in_filters=32,out_filters=64,stride=2)
        )

        self.darknet_blocks1 = self._make_layer(filters=64, blocks=n_darknet_blocks[0])
        self.down_sample1 = conv3X3(in_filters=64,out_filters=128,stride=2)
        self.darknet_blocks2 = self._make_layer(filters=128, blocks=n_darknet_blocks[1])
        self.down_sample2 = conv3X3(in_filters=128, out_filters=256, stride=2)
        self.darknet_blocks3 = self._make_layer(filters=256, blocks=n_darknet_blocks[2])
        self.down_sample3 = conv3X3(in_filters=256, out_filters=512, stride=2)
        self.darknet_blocks4 = self._make_layer(filters=512, blocks=n_darknet_blocks[3])
        self.down_sample4 = conv3X3(in_filters=512, out_filters=1024, stride=2)
        self.darknet_blocks5 = self._make_layer(filters=1024, blocks=n_darknet_blocks[4])

        #self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # output_size에 맞게 자동으로 kernel크기 설정
        #self.fc = nn.Linear(1024, classes)

    def forward(self,input):
        x = self.conv1(input)
        x = self.darknet_blocks1(x)
        x = self.down_sample1(x)
        x = self.darknet_blocks2(x)
        x = self.down_sample2(x)
        x = self.darknet_blocks3(x)
        x = self.down_sample3(x)
        x = self.darknet_blocks4(x)
        x = self.down_sample4(x)
        feature = self.darknet_blocks5(x)
        #x = self.avg_pool(x)
        #x = torch.flatten(x,1)
        #out = self.fc(x)
        return feature

    def _make_layer(self,filters,blocks):
        layers = []
        for i in range(blocks):
            layers.append(darknet_block(filters))
        return nn.Sequential(*layers)

class darknet_53_class(nn.Module):
    def __init__(self,channel=3,n_darknet_blocks=[1,2,8,8,4],classes=1000):
        super(darknet_53_class, self).__init__()
        self.classes = classes
        self.featureExtract = darknet_53(channel=channel,n_darknet_blocks=n_darknet_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # output_size에 맞게 자동으로 kernel크기 설정
        self.fc = nn.Linear(1024, classes)

    def forward(self,input):
        feature = self.featureExtract(input)
        x = self.avg_pool(feature)
        x = torch.flatten(x,1)
        out = self.fc(x)

        return out


def get_darknet_53():
    return darknet_53_class(3)

def darknet_53_featureExtract():
    return darknet_53(3)