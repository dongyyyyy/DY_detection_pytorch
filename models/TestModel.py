from torchsummary import summary
from models.ResNet import *
from models.darknet_53 import *

if __name__ == '__main__':
    resnet18 = get_resnet18()
    summary(resnet18.cuda(), input_size=(3, 224, 224))
    resnet50 = get_resnet50()
    summary(resnet50.cuda(), input_size=(3, 224, 224))
    '''
    darknet = get_darknet_53()
    summary(darknet.cuda(), input_size=(3,416,416))
    '''
