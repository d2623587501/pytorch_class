from config import config


def model_config():
    if 'alexnet' in config.model:
        if config.model == 'alexnet':
            from model_zoo.alexNet.alexNet import AlexNet
            return AlexNet
        else:
            print('请选取正确的alexnet模型范围')

    elif 'vggnet' in config.model:
        if config.model == 'vggnet':
            from model_zoo.vggNet.vggNet import vgg
            return vgg
        else:
            print('请选取正确的vggnet模型范围')

    elif 'resnet' in config.model:
        if config.model == 'resnet34':
            from model_zoo.resNet.resNet import resnet34
            return resnet34
        if config.model == 'resnet50':
            from model_zoo.resNet.resNet import resnet50
            return resnet50
        if config.model == 'resnet101':
            from model_zoo.resNet.resNet import resnet101
            return resnet101
        if config.model == 'resnext50_32x4d':
            from model_zoo.resNet.resNet import resnext50_32x4d
            return resnext50_32x4d
        if config.model == 'resnext101_32x8d':
            from model_zoo.resNet.resNet import resnext101_32x8d
            return resnext101_32x8d
        else:
            print('请选取正确的resnet模型范围')

    elif 'regnet' in config.model:
        if config.model == 'regnet':
            from model_zoo.regNet.regNet import create_regnet
            return create_regnet
        else:
            print('请选取正确的regnet模型范围')

    elif 'mobilenet' in config.model:
        if config.model == 'mobilenet_v2':
            from model_zoo.mobileNet.mobileNet_v2 import MobileNetV2
            return MobileNetV2
        else:
            print('请选取正确的mobilenet模型范围')

    elif 'densenet' in config.model:
        if config.model == 'densenet121':
            from model_zoo.denseNet.denseNet import densenet121
            return densenet121
        if config.model == 'densenet161':
            from model_zoo.denseNet.denseNet import densenet161
            return densenet161
        if config.model == 'densenet169':
            from model_zoo.denseNet.denseNet import densenet169
            return densenet169
        if config.model == 'densenet201':
            from model_zoo.denseNet.denseNet import densenet201
            return densenet201
        else:
            print('请选取正确的densenet模型范围')

    elif 'shuffle' in config.model:
        if config.model == 'shufflenet_v2_x0_5':
            from model_zoo.shuffleNet.shuffleNet import shufflenet_v2_x0_5
            return shufflenet_v2_x0_5
        if config.model == 'shufflenet_v2_x1_0':
            from model_zoo.shuffleNet.shuffleNet import shufflenet_v2_x1_0
            return shufflenet_v2_x1_0
        if config.model == 'shufflenet_v2_x1_5':
            from model_zoo.shuffleNet.shuffleNet import shufflenet_v2_x1_5
            return shufflenet_v2_x1_5
        if config.model == 'shufflenet_v2_x1_0':
            from model_zoo.shuffleNet.shuffleNet import shufflenet_v2_x2_0
            return shufflenet_v2_x2_0
        else:
            print('请选取正确的shufflenet模型范围')

    elif 'efficientnet' in config.model:
        if config.model == 'efficientnet_b0':
            from model_zoo.efficientNet.efficientNet import efficientnet_b0
            return efficientnet_b0
        elif config.model == 'efficientnet_b1':
            from model_zoo.efficientNet.efficientNet import efficientnet_b1
            return efficientnet_b1
        elif config.model == 'efficientnet_b2':
            from model_zoo.efficientNet.efficientNet import efficientnet_b2
            return efficientnet_b2
        elif config.model == 'efficientnet_b3':
            from model_zoo.efficientNet.efficientNet import efficientnet_b3
            return efficientnet_b3
        elif config.model == 'efficientnet_b4':
            from model_zoo.efficientNet.efficientNet import efficientnet_b4
            return efficientnet_b4
        elif config.model == 'efficientnet_b5':
            from model_zoo.efficientNet.efficientNet import efficientnet_b5
            return efficientnet_b5
        elif config.model == 'efficientnet_b6':
            from model_zoo.efficientNet.efficientNet import efficientnet_b6
            return efficientnet_b6
        elif config.model == 'efficientnet_b7':
            from model_zoo.efficientNet.efficientNet import efficientnet_b7
            return efficientnet_b7
        elif config.model == 'efficientnetv2_s':
            from model_zoo.efficientNet.efficientNet_v2 import efficientnetv2_s
            return efficientnetv2_s
        elif config.model == 'efficientnetv2_l':
            from model_zoo.efficientNet.efficientNet_v2 import efficientnetv2_l
            return efficientnetv2_l
        elif config.model == 'efficientnetv2_m':
            from model_zoo.efficientNet.efficientNet_v2 import efficientnetv2_m
            return efficientnetv2_m
        else:
            print('请选取正确的efficientnet模型范围')
    else:
        print('请选取范围的模型')
