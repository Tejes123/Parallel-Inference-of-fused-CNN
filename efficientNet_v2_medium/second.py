# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
from torch import nn
import numpy as np
from model_partition import get_second_stage_efficientNet
from fuse_models import fuse_resnet
import zmq, pickle
import tensorrt as trt
import time
from onnx_file import engine_infer_single_batch

# Model without layer fuion without tensorrt 
PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_1 = r"stage_1/efficientNet_no_layer_no_tensorrt_1.pth"

# Model without fusion and with tensorrt
PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_1 = r"stage_1/efficientNet_no_layer_tensorrt_1.trt"

# Model with Fusion and witout Tensorrt 
PATH_RESNET_LAYER_NO_TENSORRT_MODEL_1 = r"stage_1/efficientNet_layer_no_tensorrt_1.pth"

# Model with Layer and with TensorRT
PATH_RESNET_LAYER_TENSORRT_ENGINE_1 = r"stage_1/efficientNet_layer_tensorrt_1.trt"

PATH = r"resnet50_fused_new.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context = zmq.Context()
socket_pc_0 = context.socket(zmq.PULL)
socket_pc_0.bind("tcp://*:25900")   # connect to PC0’s push socket

socket_stage_2= context.socket(zmq.PUSH)
socket_stage_2.connect("tcp://127.0.0.1:3000")   # connect to PC0’s push socket

socket_pc_0.setsockopt(zmq.SNDBUF, 10* 1024)
socket_pc_0.setsockopt(zmq.RCVBUF, 10 * 1024 )

socket_stage_2.setsockopt(zmq.SNDBUF, 10 * 1024)
socket_stage_2.setsockopt(zmq.RCVBUF, 10 * 1024 )

# print(int(socket_stage0.getsockopt(zmq.SNDBUF)))
# print(int(socket_stage0.getsockopt(zmq.RCVBUF)))
# print(int(socket_stage_2.getsockopt(zmq.SNDBUF)))
# print(int(socket_stage_2.getsockopt(zmq.RCVBUF)))

def without_fusion_without_tensorrt():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = torchvision.models.efficientnet_v2_m()

    stage_1 = get_second_stage_efficientNet(model)
    
    stage_1.load_state_dict(torch.load(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_1, weights_only = True))
    stage_1  = stage_1.to(device)
    total_time = 0
    total_execution_time = 0
    start_inference = time.time()

    while True:
        data = socket_pc_0.recv() 
        
        start_execution_time = time.time()
        activations = pickle.loads(data)

        # activations = torch.rand(16, 1024, 2, 2)
        
        activations = activations.to(device)
        
        batch_start_time = time.time()  
        final_output = stage_1(activations)
        batch_end_time = time.time()  
        
        final_output = final_output.to('cpu')
        
        total_time += (batch_end_time - batch_start_time)

        batch_count += 1
        print(f"Batch {batch_count}: arrived at PC1. Total Time Taken : {(batch_end_time - batch_start_time) * 1000 :.2f} ms")

        # Send to the third stage 
        final_output_serialized = pickle.dumps(final_output)
        socket_stage_2.send(final_output_serialized)
        print(f"Batch {batch_count}: Sent to PC2")

        stop_execution_time = time.time()
        total_execution_time += (stop_execution_time - start_execution_time)

        # Temperorary
        if(batch_count == 624):
            print(f"Total Inference Time Taken : {total_time*1000:.3f} ms")
            print(f" Total Execution Time Taken : {total_execution_time *1000:.3f} ms")
            break


def without_fusion_with_tensorrt():
    batch_count = 0
    outputs = [] 

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_1, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    total_time = 0
    total_execution_time = 0
    batch_count = 0

    while True:
        data = socket_pc_0.recv() 
        start_execution_time = time.time()

        activations = pickle.loads(data)
        # activations = torch.rand(16, 512, 4, 4)

        batch_start_time = time.time() 
        final_output = engine_infer_single_batch(engine, context, activations, batch_count=16)
        batch_end_time = time.time()         

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1

        print(f"Batch {batch_count} | Total Time Taken: {(batch_end_time - batch_start_time)*1000:.3f} ms" )
        

        # Send to the third stage 
        final_output_serialized = pickle.dumps(final_output)
        socket_stage_2.send(final_output_serialized)
        print(f"Batch {batch_count}: Sent to PC2")

        end_execution = time.time()
        total_execution_time += (end_execution -start_execution_time)

        if(batch_count == 624):
            print(f"Total Time Taken for all Batches in stage 1: {(total_time)*1000:.3f} ms")
            print(f" Total Execution Time Taken : {total_execution_time *1000:.3f} ms")
            break


def with_fuson_without_tensorrt():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = torchvision.models.efficientnet_v2_m()

    stage_1 = get_second_stage_efficientNet(model)

    # Fuse the model
    stage_1 = fuse_resnet(stage_1)

    stage_1.load_state_dict(torch.load(PATH_RESNET_LAYER_NO_TENSORRT_MODEL_1, weights_only = True))
    stage_1 = stage_1.to(device)
    total_time = 0
    total_execution_time = 0

    print("Waiting For PC0 to send activations")
    while True:
        activations = socket_pc_0.recv() 

        start_esecution_time = time.time()
        
        activations = pickle.loads(activations)
        activations = activations.to(device) 

        batch_start_time = time.time() 
        final_output = stage_1(activations) 
        batch_end_time = time.time()

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        print(f"Batch {batch_count} : Total Time: {(batch_end_time - batch_start_time) * 1000:.3f}")
    
        # Send to the third stage 
        final_output_serialized = pickle.dumps(final_output)
        socket_stage_2.send(final_output_serialized)
        print(f"Batch {batch_count}: Sent to PC2")

        end_execution_time = time.time()
        total_execution_time += (end_execution_time - start_esecution_time)

        if(batch_count == 624):
            print(f"Total Time Taken for {batch_count} batches: {total_time*1000:.2f} ms")
            print(f" Total Execution Time Taken : {total_execution_time *1000:.3f} ms")
            break

def with_fusion_with_tensorrt():
    batch_count = 0
    outputs = [] 

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_LAYER_TENSORRT_ENGINE_1, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    total_time = 0
    total_execution_time = 0
    batch_count = 0

    while True:
        data = socket_pc_0.recv()    
        start_execution_time =time.time()
    
        activations = pickle.loads(data)

        # activations = torch.rand(16, 512, 4, 4)
        batch_start_time = time.time() 
        final_output = engine_infer_single_batch(engine, context, activations, batch_count=16)
        batch_end_time = time.time()     

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1

        print(f"Batch {batch_count} | Total Time Taken: {(batch_end_time - batch_start_time)*1000:.3f} ms" )

        end_execution_time = time.time()
        total_execution_time += (end_execution_time - start_execution_time)

         # Send to the third stage 
        final_output_serialized = pickle.dumps(final_output)
        socket_stage_2.send(final_output_serialized)
        print(f"Batch {batch_count}: Sent to PC2")

        if(batch_count == 624):
            print(f"Total Time Taken for all Batches in stage 1: {(total_time)*1000:.3f} ms")
            print(f" Total Execution Time Taken : {total_execution_time *1000:.3f} ms")

            break


# without_fusion_without_tensorrt()
# without_fusion_with_tensorrt()
# with_fuson_without_tensorrt()
with_fusion_with_tensorrt()