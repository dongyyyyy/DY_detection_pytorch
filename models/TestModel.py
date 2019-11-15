from models.ResNet import *

if __name__ == '__main__':
    resnet18 = ResNet(input_shape=3, n_residual_blocks=[2, 2, 2, 2], basic_block='v1')
    summary(resnet18.cuda(), input_size=(3, 224, 224))
    resnet101 = ResNet(input_shape=3, n_residual_blocks=[3, 4, 3, 3], basic_block='v2')
    summary(resnet101.cuda(), input_size=(3, 224, 224))
