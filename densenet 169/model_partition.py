import torchvision.models as models 
import torch.nn as nn 
from torchvision.models.densenet import _DenseBlock

densenet_model = models.densenet169()

model = densenet_model.features
classifier = densenet_model.classifier

def get_first_stage_densenet(densenet):
    stage0 = nn.Sequential()

    stage0_layer3_1 = _DenseBlock(num_layers=9, num_input_features=256, growth_rate=32, drop_rate=0, bn_size=4)

    # Add for stage 0
    stage0.add_module("conv0", model.conv0)
    stage0.add_module("norm0", model.norm0)
    stage0.add_module("relu0", model.relu0)
    stage0.add_module("pool0", model.pool0)

    stage0.add_module("denseblock1", model.denseblock1)
    stage0.add_module("transition1", model.transition1)

    stage0.add_module("denseblock2", model.denseblock2)
    stage0.add_module("trasition2", model.transition2)

    stage0.add_module("custom_denseblock3_stage0", stage0_layer3_1)

    return stage0

def get_second_stage_densenet(densenet):
    stage1 = nn.Sequential()

    stage1_layer3 = _DenseBlock(num_layers=23, num_input_features=544, bn_size=4, growth_rate=32, drop_rate=0)
    stage1_transtition3 = model.transition3 
    stage1_layer4 = _DenseBlock(num_layers=5, num_input_features=640, bn_size=4, growth_rate=32, drop_rate=0)

    stage1.add_module("customm_denseblock3_stage1", stage1_layer3)
    stage1.add_module("transiton3", stage1_transtition3)
    stage1.add_module("custom_denseblock4_stage1", stage1_layer4)

    return stage1  

def get_third_stage_densenet(densenet):
    stage2= nn.Sequential()

    stage2_layer4 = _DenseBlock(num_layers = 27, num_input_features=800, bn_size=4, growth_rate=32, drop_rate=0)

    stage2.add_module("custom_denseblock_stage2", stage2_layer4)
    stage2.add_module("norm5", model.norm5)
    stage2.add_module("custm_relu", nn.ReLU(inplace=True) )
    stage2.add_module("custm_adp_avg_pool_2d", nn.AdaptiveAvgPool2d((1, 1)))
    stage2.add_module("custm_flatten", nn.Flatten(1))
    stage2.add_module("classifier", classifier)

    return stage2
