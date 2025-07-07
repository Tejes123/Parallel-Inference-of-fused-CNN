# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
from torchvision.models import resnet50
from torch import nn
from fuse_models import fuse_resnet
from data_loader import get_complete_loader, get_test_loader
from model_partition import get_first_stage_resnet, get_second_stage_resnet

# a Function that returns the modified Resnet50 model for CIFAR 10 dataset - last layer has size 10 
def get_resnet50_for_CIFAR10():
    # CIFAR 10 has 10 classes, so the last layer should be a fully conneted with size 10
    net = torchvision.models.resnet50()

    # Get the number of input features to the last fully connected layer
    in_features = net.fc.in_features

    # Replace the last fully connected layer with a new one for 10 classes
    net.fc = nn.Linear(in_features, 10)
    return net

model = get_resnet50_for_CIFAR10().eval()

# Fuse the model
fused_model = fuse_resnet(model)

# Partition the models
stage0 = get_first_stage_resnet(fused_model)
# stage1 = get_second_stage_resnet(fused_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ pc0.py ------------
#Move stage0 tp GPU
stage0 = stage0.to(device)

import zmq, pickle, time
import torch

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://10.10.5.156:25900")     # PC0 pushes on port 5555

# dummy data loader
data_loader = get_test_loader(batch_size = 16)

print("Move Stage 0 to GPU")
stage0 = stage0.to(device)

for(i, batch) in enumerate(data_loader):
    data, label = batch
    # print(data)
    # Move the input data tensor to GPU
    data = data.to(device)
    print(f"Batch: {i} | Movied input data to GPU")

    # Compyute the activation
    activations = stage0(data)
    print(f"Batch: {i} | Computated activations")

    # Move the activation to CPU
    activation_serialized = pickle.dumps(activations.cpu())
    # print(f"Batch: {i} | Moved activation to CPU ")

    socket.send(activation_serialized)
    print(f"Batch: {i} | Activatiosn sent to PC1")