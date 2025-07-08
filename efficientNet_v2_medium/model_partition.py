import torch.nn as nn 
import torchvision 

efficientnet = torchvision.models.efficientnet_v2_m()

model = efficientnet.features
e_avg_pool = efficientnet.avgpool
e_classifier = efficientnet.classifier

children_list = list(model.children())
classifier_in_features = e_classifier[1].in_features

def get_first_stage_efficientNet(efficientnet):
    model = efficientnet.features

    children_list = list(model.children())

    stage0 = nn.Sequential(
        children_list[0],
        children_list[1],
        children_list[2],
        children_list[3],
        children_list[4],
        nn.Sequential(*children_list[5][0:5])
    )

    return stage0

def get_second_stage_efficientNet(efficientnet):
    model = efficientnet.features
    # e_avg_pool = efficientnet.avgpool
    # e_classifier = efficientnet.classifier

    children_list = list(model.children())

    stage1 = nn.Sequential(
        nn.Sequential(*children_list[5][5:]),
        nn.Sequential(*children_list[6][0:9]) # continue    
    )

    return stage1  

def get_third_stage_efficientNet(efficientnet):
    model = efficientnet.features
    e_avg_pool = efficientnet.avgpool
    # e_classifier = efficientnet.classifier

    children_list = list(model.children())

    stage2 = nn.Sequential(
        nn.Sequential(*list(children_list[6][9:])),
        children_list[7],
        children_list[8], # Conv2dNormActivation
        e_avg_pool,
        nn.Flatten(1),
        nn.Dropout1d(p = 0.2, inplace=True),
        nn.Linear(in_features= classifier_in_features, out_features=10)
        # e_classifier
    )

    return stage2  