import torch.nn as nn

def get_first_stage_resnet(model):
    stage1 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        *list(model.layer3.children())[0:5]
    )
    return stage1

    
def get_second_stage_resnet(model):
    # The input shape of tensor for the second stage is: 
    stage2 = nn.Sequential(
        *list(model.layer3.children())[5:22]
    )
    return stage2

def get_third_stage_resnet(model):
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, 10)
    
    stage3 = nn.Sequential(
        *list(model.layer3.children())[22:],  # remaining 25 bottlenecks
        model.layer4,
        model.avgpool,
        nn.Flatten(),
        model.fc
    )
    return stage3 