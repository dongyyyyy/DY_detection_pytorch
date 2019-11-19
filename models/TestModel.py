from torchsummary import summary
from models.ResNet import *
from models.darknet_53 import *
from models.VGGNet import *
from models.Hourglass import *
if __name__ == '__main__':
    '''
    resnet_extract1 = ResNet(input_shape=3,n_residual_blocks=[2,2,2,2],basic_block=BottleNeck)
    resnet_extract2 = ResNet(input_shape=3,n_residual_blocks=[2,2,2,2],basic_block=BasicBlock)
    summary(resnet_extract1.cuda(),input_size=(3,224,224))
    summary(resnet_extract2.cuda(),input_size=(3,224,224))
    print('='*50)
    resnet18 = get_resnet18()
    summary(resnet18.cuda(), input_size=(3, 224, 224))
    resnet50_class = get_resnet50()
    summary(resnet50_class.cuda(), input_size=(3, 224, 224))
    

    darknet = darknet_53()
    summary(darknet.cuda(), input_size=(3,416,416))
    '''
#    VGG = VGG_class(input_shape=3,layer="VGG16")
#    summary(VGG.cuda(),input_size=(3,224,224))

    #VGG = VGG_class(input_shape=3, layer="VGG19")
    #summary(VGG.cuda(), input_size=(3, 224, 224))

    summary(Hourglass(input_shape=3).cuda(), input_size=(3, 256, 256))


