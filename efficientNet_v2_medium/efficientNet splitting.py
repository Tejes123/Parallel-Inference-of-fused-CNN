import torch.nn as nn 
import torchvision 

efficientnet = torchvision.models.efficientnet_v2_s()

model = efficientnet.features
avg_pool = efficientnet.avgpool
classifier = efficientnet.classifier

print(list(model.children())[5][1])

children_list = list(model.children())


stage0 = nn.Sequential(
    children_list[0],
    children_list[1],
    children_list[2],
    children_list[3],
    children_list[4],
    nn.Sequential(*children_list[5][0:2])
)

stage1 = nn.Sequential(
    nn.Sequential(*children_list[5][2:]),
    # nn.Sequential(*children_list[6][]) # continue
)

print(stage1)