# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
from torch import nn
from model_partition import get_first_stage_resnet, get_second_stage_resnet
from fuse_models import fuse_resnet
from data_loader import get_complete_loader, get_test_loader
import zmq, pickle
import tensorrt as trt
import time
from onnx_file import engine_infer_single_batch

# Model without layer fuion without tensorrt 
PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_ONE = r"resnet_no_layer_no_tensorrt_1.pth"

# Model without fusion and with tensorrt
PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_ONE = r"resnet_no_layer_tensorrt_1.trt"

# Model with Fusion and witout Tensorrt 
PATH_RESNET_LAYER_NO_TENSORRT_MODEL_ONE = r"resnet_layer_no_tensorrt_1.pth"

# Model with Layer and with TensorRT
PATH_RESNET_LAYER_TENSORRT_ENGINE_ONE = r"resnet_layer_tensorrt_1.trt"

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


def imp():
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
        out = out.to('cpu')                     
        time.sleep(0.5) 
        print(f"Batch: {curr_batch}: Computation Comppleted")
        outputs.append(out)
        curr_batch += 1

    final = torch.cat(outputs, dim=0)
    print("PC1: got final output of shape", final.shape)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:25900")   # connect to PC0’s push socket

def without_fusion_without_tensorrt():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = get_resnet50_for_CIFAR10()
    stage_1 = get_second_stage_resnet(model)
    stage_1.load_state_dict(torch.load(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_ONE, weights_only = True))
    stage_1  = stage_1.to(device)
    total_time = 0

    while True:
        data = socket.recv() 
        activations = pickle.loads(data)
        
        activations = activations.to(device)
        batch_start_time = time.time()  
        final_output = stage_1(activations)
        batch_end_time = time.time()  
        final_output = final_output.to('cpu')
        
        total_time += (batch_end_time - batch_start_time)

        outputs.append(final_output)
        batch_count += 1
        print(f"Batch {batch_count}: arrived at PC1. Total Time Taken : {(batch_end_time - batch_start_time) * 1000 :.2f} ms")

        # Temperorary
        if(batch_count == 624):
            print(f"Total Tim Taken : {total_time*1000:.3f} ms")
            break


def without_fusion_with_tensorrt():
    batch_count = 0
    outputs = [] 

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_ONE, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    total_time = 0
    batch_count = 0

    while True:
        data = socket.recv() 
        batch_start_time = time.time() 
        activations = pickle.loads(data)
        # activations = torch.rand(16, 512, 4, 4)
        final_output = engine_infer_single_batch(engine, context, activations, batch_count=16)
        batch_end_time = time.time()          
        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        outputs.append(final_output)
        print(f"Batch {batch_count} | Total Time Taken: {(batch_end_time - batch_start_time)*1000:.3f} ms" )

        if(batch_count == 624):
            print(f"Total Time Taken for all Batches in stage 1: {(total_time)*1000:.3f} ms")
            break


def with_fuson_without_tensorrt():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = get_resnet50_for_CIFAR10()
    stage_1 = get_second_stage_resnet(model)

    # Fuse the model
    stage_1 = fuse_resnet(stage_1)

    stage_1.load_state_dict(torch.load(PATH_RESNET_LAYER_NO_TENSORRT_MODEL_ONE, weights_only = True))
    stage_1 = stage_1.to(device)
    total_time = 0
    print("Waiting For PC0 to send activations")
    while True:
        data = socket.recv() 
        batch_start_time = time.time() 
        activations = pickle.loads(data)

        activations = activations.to(device) 
        final_output = stage_1(activations) 
        batch_end_time = time.time()

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        outputs.append(final_output)
        print(f"Batch {batch_count} : Total Time: {(batch_end_time - batch_start_time) * 1000:.3f}")
    
        if(batch_count == 624):
            print(f"Total Time Taken for {batch_count} batches: {total_time*1000:.2f} ms")
            break

def with_fusion_with_tensorrt():
    batch_count = 0
    outputs = [] 

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_LAYER_TENSORRT_ENGINE_ONE, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    total_time = 0
    batch_count = 0

    while True:
        data = socket.recv() 
        batch_start_time = time.time() 
        activations = pickle.loads(data)
        # activations = torch.rand(16, 512, 4, 4)
        final_output = engine_infer_single_batch(engine, context, activations, batch_count=16)
        batch_end_time = time.time()          
        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        outputs.append(final_output)
        print(f"Batch {batch_count} | Total Time Taken: {(batch_end_time - batch_start_time)*1000:.3f} ms" )

        if(batch_count == 624):
            print(f"Total Time Taken for all Batches in stage 1: {(total_time)*1000:.3f} ms")
            break

# without_fusion_without_tensorrt()
# without_fusion_with_tensorrt()
# with_fuson_without_tensorrt()
with_fusion_with_tensorrt()