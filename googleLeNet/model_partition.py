import torchvision 
import torch.nn as nn 

# model = torchvision.models.GoogLeNet()


def get_first_stage_googleLeNet(model):
    stage0 = nn.Sequential(
        model.conv1,
        model.maxpool1,
        model.conv2,
        model.conv3,
        model.maxpool2,

        model.inception3a,
        model.inception3b,
        model.maxpool3,
        model.inception4a
    )

    return stage0

def get_second_stage_googleLeNet(model):
    stage1 = nn.Sequential(
        model.inception4b,
        model.inception4c,
        model.inception4d,
        model.inception4e,
        model.maxpool4,
    )

    return stage1  

def get_third_stage_googleLeNet(model):

    stage2 = nn.Sequential(
        model.inception5a,
        model.inception5b,
        # model.aux1, 
        # model.aux2,
        model.avgpool,
        nn.Flatten(1),
        model.dropout,
        model.fc
    )

    return stage2  