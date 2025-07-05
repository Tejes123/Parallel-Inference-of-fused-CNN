# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
import pycuda.driver as cuda
import numpy as np
import time
import tensorrt as trt
import zmq, pickle, time
import torch

from torchvision.models import resnet50
from torch import nn
from fuse_models import fuse_resnet
from data_loader import get_complete_loader, get_test_loader
from model_partition import get_first_stage_resnet, get_second_stage_resnet
from onnx_file import engine_infer_single_batch

# Model without layer fuion without tensorrt 
PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_0 = r"resnet_no_layer_no_tensorrt_0.pth"

# Model without fusion and with tensorrt
PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_0 = r"resnet_no_layer_tensorrt_0.trt"

# Model with Fusion and witout Tensorrt 
PATH_RESNET_LAYER_NO_TENSORRT_0 = r"resnet_layer_no_tensorrt_0.pth"

# Model with Layer and with TensorRT
PATH_RESNET_LAYER_TENSORRT_ENGINE_0 = r"resnet_layer_tensorrt_0.trt"

# a Function that returns the modified Resnet50 model for CIFAR 10 dataset - last layer has size 10 
def get_resnet50_for_CIFAR10():
    # CIFAR 10 has 10 classes, so the last layer should be a fully conneted with size 10
    net = torchvision.models.resnet50()

    # Get the number of input features to the last fully connected layer
    in_features = net.fc.in_features

    # Replace the last fully connected layer with a new one for 10 classes
    net.fc = nn.Linear(in_features, 10)
    return net

#     return host_output  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://10.10.5.156:25900")     # PC0 pushes on port 5555
data_loader = get_test_loader(batch_size = 16)

def imp():
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

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://10.10.0.208:25900")     # PC0 pushes on port 5555

    # dummy data loader
    data_loader = get_test_loader(batch_size = 16)
    print(len(data_loader))

    print("Move Stage 0 to GPU")
    stage0 = stage0.to(device)

    batch_count = len(data_loader)

def without_fusion_without_tensorrt():
    model = get_resnet50_for_CIFAR10().eval()
    part_0 = get_first_stage_resnet(model)
    part_0.load_state_dict(torch.load(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_0, weights_only=True))  
    stage_0 = part_0.to(device)
    total_time = 0

    for (i, batch) in enumerate(data_loader):
        start_time = time.time()
        (input_data, label) = batch
        input_data = input_data.to(device)
        output_activation = stage_0(input_data)
        output_activation = output_activation.to("cpu")
        serialized_act = pickle.dumps(output_activation)
        socket.send(serialized_act)
        batch_time = time.time() - start_time
        print(f"Batch {i} : Sent | Time Taken: {batch_time * 1000:.3f} ms")
        total_time += batch_time

    print(f"Total Time Taken: {total_time * 1000:.3f} ms")
    socket.close()


def without_fusion_with_tensorrt():

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_0, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    total_time = 0
    for(i, batch) in enumerate(data_loader):
        start_time = time.time()
        (input_data, label) = batch 
        activation = engine_infer_single_batch(engine, context, input_data, i)
        activation_serialized = pickle.dumps(activation)
        socket.send(activation_serialized)
        end_time = time.time() 

        batch_time = end_time - start_time 
        print(f"Batch {i} : Sent | Time Taken: {batch_time * 1000:.3f} ms")
        total_time += batch_time
    print(f"Total time taken: {total_time * 1000:.3f} ms")
    
def with_fusion_without_tensorrt():
    model = get_resnet50_for_CIFAR10().eval()
    part_0 = get_first_stage_resnet(model)

    # Fuse the stage 0
    fused_part_0 = fuse_resnet(part_0)

    fused_part_0.load_state_dict(torch.load(PATH_RESNET_LAYER_NO_TENSORRT_0, weights_only=True))  
    fused_part_0 = fused_part_0.to(device)

    total_time = 0

    for (i, batch) in enumerate(data_loader):
        start_time = time.time()
        (input_data, label) = batch
        input_data = input_data.to(device)

        output_activation = fused_part_0(input_data)
        serialized_act = pickle.dumps(output_activation)

        socket.send(serialized_act)
        batch_time = time.time() - start_time

        print(f"Batch {i} : Sent | Time Taken: {batch_time * 1000:.3f} ms")
        total_time += batch_time

    print(f"Total Time Taken: {total_time * 1000:.3f} ms")

def with_fusion_with_tensorrt():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_LAYER_TENSORRT_ENGINE_0, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    total_time = 0
    for(i, batch) in enumerate(data_loader):
        start_time = time.time()
        (input_data, label) = batch 
        activation = engine_infer_single_batch(engine, context, input_data, i)
        activation_serialized = pickle.dumps(activation)
        end_time = time.time()
        socket.send(activation_serialized) 

        batch_time = end_time - start_time 
        print(f"Batch {i} : Sent | Time Taken: {batch_time * 1000:.3f} ms")
        total_time += batch_time
    print(f"Total time taken: {total_time * 1000:.3f} ms")    

# without_fusion_without_tensorrt()
# without_fusion_with_tensorrt()
# with_fusion_without_tensorrt()
with_fusion_with_tensorrt() 