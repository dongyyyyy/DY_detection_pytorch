import torch.nn as nn
import torch

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

class VGG_backbone(nn.Module):
    def __init__(self, input_shape=3, layer="VGG16"):
        super(VGG_backbone, self).__init__()
        if layer == "VGG16":
            self.block_num = 3
        elif layer == "VGG19":
            self.block_num = 4
        else:
            self.block_num = 0

        self.layer1 = nn.Sequential(
            conv3X3(in_filters=input_shape,out_filters=64),
            conv3X3(in_filters=64, out_filters=64),
        )

        self.layer2 = nn.Sequential(
            conv3X3(in_filters=64, out_filters=128),
            conv3X3(in_filters=128, out_filters=128),
        )

        self.layer3 = self._make_model(in_filters=128,out_filters=256,block_num=self.block_num)

        self.layer4 = self._make_model(in_filters=256, out_filters=512, block_num=self.block_num)

        self.layer5 = self._make_model(in_filters=512, out_filters=512, block_num=self.block_num)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self,input):
        x = self.layer1(input)
        x = self.max_pool(x) # max_pool의 경우에는 Sequential에 넣을 경우 중복되서 사용되는 경우가 존재함
        x = self.layer2(x)
        x = self.max_pool(x)
        x = self.layer3(x)
        x = self.max_pool(x)
        x = self.layer4(x)
        x = self.max_pool(x)
        x = self.layer5(x)
        out = self.max_pool(x)
        print(out.shape)
        return out

    def _make_model(self,in_filters,out_filters,block_num=3):
        layers = []
        for i in range(block_num):
            if i == 0:
                layers.append(conv3X3(in_filters=in_filters,out_filters=out_filters))
            else:
                layers.append(conv3X3(in_filters=out_filters,out_filters=out_filters))

        return nn.Sequential(*layers)

class VGG_class(nn.Module):
    def __init__(self,input_shape=3,layer="VGG16",classes = 1000):
        super(VGG_class, self).__init__()
        self.FeatureExtract = VGG_backbone(input_shape=3,layer=layer)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

    def forward(self,input):
        x = self.FeatureExtract(input)
        x = torch.flatten(x,1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out