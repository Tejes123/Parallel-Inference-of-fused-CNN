import torch.nn as nn

def get_first_stage_resnet(model):
    stage1 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu, 
        model.maxpool,
        model.layer1,
        model.layer2
    )
    return stage1

    
def get_second_stage_resnet(model):
    # The input shape of tensor for the second stage is: 
    # torch.Size([16, 512, 4, 4]), where 16 = batch_size 
    stage2 = nn.Sequential(
        model.layer3,
        model.layer4,
        model.avgpool,
        nn.Flatten(),
        model.fc
    )
    return stage2