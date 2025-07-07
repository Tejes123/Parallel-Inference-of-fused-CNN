# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
from torchvision.models import resnet50
from torch import nn
from model_partition import get_first_stage_resnet, get_second_stage_resnet
from fuse_models import fuse_resnet
from data_loader import get_complete_loader
import zmq, pickle
import time

PATH = r"resnet50_fused_new.pth"

# a Function that returns the modified Resnet50 model for CIFAR 10 dataset - last layer has size 10 
def get_resnet50_for_CIFAR10():
    # CIFAR 10 has 10 classes, so the last layer should be a fully conneted with size 10
    net = torchvision.models.resnet50()

    # Get the number of input features to the last fully connected layer
    in_features = net.fc.in_features

    # Replace the last fully connected layer with a new one for 10 classes
    net.fc = nn.Linear(in_features, 10)
    return net

model = get_resnet50_for_CIFAR10()

# Fuse the model
fused_model = fuse_resnet(model)

# Optional
# fused_model.load_state_dict(torch.load(PATH, weights_only=True))


# Partition the models
stage1 = get_second_stage_resnet(fused_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stage1 = stage1.to(device)

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:25900")   # connect to PC0’s push socket

outputs = []
curr_batch = 0

while True:
    data = socket.recv()
    print(f"Batch {curr_batch}: arrived at PC1")

    # deserialize to Tensor
    activation = pickle.loads(data) 
    # run the rest of ResNet
    activation = activation.to(device)

    out = stage1(activation)                     
    # time.sleep(1) 
    print(f"Batch: {curr_batch}: Computation Comppleted")
    outputs.append(out.cpu())
    curr_batch += 1

final = torch.cat(outputs, dim=0)
print("PC1: got final output of shape", final.shape)