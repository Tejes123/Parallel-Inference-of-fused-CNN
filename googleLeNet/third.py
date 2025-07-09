# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
from torch import nn
from model_partition import get_third_stage_googleLeNet, get_first_stage_googleLeNet, get_second_stage_googleLeNet
from fuse_models import fuse_resnet
from data_loader import get_complete_loader, get_test_loader
import zmq, pickle
import tensorrt as trt
import time
from onnx_file import engine_infer_single_batch

# Model without layer fuion without tensorrt 
PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_2 = r"stage_2/googlelenet_no_layer_no_tensorrt_2.pth"

# Model without fusion and with tensorrt
PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_2 = r"stage_2/googlelenet_no_layer_tensorrt_2.trt"

# Model with Fusion and witout Tensorrt 
PATH_RESNET_LAYER_NO_TENSORRT_MODEL_2 = r"stage_2/googlelenet_layer_no_tensorrt_2.pth"

# Model with Layer and with TensorRT
PATH_RESNET_LAYER_TENSORRT_ENGINE_2 = r"stage_2/googlelenet_layer_tensorrt_2.trt"

PATH = r"resnet50_fused_new.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://127.0.0.1:3000")   # connect to PC0’s push socket

socket.setsockopt(zmq.SNDBUF, 10  * 1024)
socket.setsockopt(zmq.SNDBUF, 10  * 1024 )

def without_fusion_without_tensorrt():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = torchvision.models.GoogLeNet()
    
    stage_1 = get_third_stage_googleLeNet(model)
    stage_1.load_state_dict(torch.load(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_2, weights_only = True))
    stage_1  = stage_1.to(device)
    total_time = 0
    total_execution_time = 0

    while True:
        data = socket.recv() 
        start_execution_time = time.time()

        activations = pickle.loads(data)
        activations = activations.to(device)
        
        batch_start_time = time.time()  
        final_output = stage_1(activations)
        batch_end_time = time.time()  
        
        final_output = final_output.to('cpu')
        
        total_time += (batch_end_time - batch_start_time)
        batch_count += 1

        print(f"Batch {batch_count}: arrived at PC3. Total Time Taken : {(batch_end_time - batch_start_time) * 1000 :.2f} ms")

        end_execution_time = time.time()
        total_execution_time += (end_execution_time - start_execution_time) # danger Zone dnt do anthing 
        # Temperorary
        if(batch_count == 624):
            print(f"Total Tim Taken in Third Stage: {total_time*1000:.3f} ms")
            print(f"End Time: {end_execution_time * 1000:.3f} ms")
            print(f"Total Execution Time: {(total_execution_time * 1000):.3f} ms")
            break

def without_fusion_with_tensorrt():
    batch_count = 0
    outputs = [] 

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_2, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    total_time = 0
    batch_count = 0
    total_execution_time = 0

    while True:
        data = socket.recv() 
        start_execution_time = time.time()
    
        activations = pickle.loads(data)

        # activations = torch.rand(16, 512, 4, 4)
        batch_start_time = time.time() 
        final_output = engine_infer_single_batch(engine, context, activations, batch_count=16)
        batch_end_time = time.time()    

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        print(f"Batch {batch_count} | Total Time Taken: {(batch_end_time - batch_start_time)*1000:.3f} ms" )

        end_execution_time = time.time()
        total_execution_time += (end_execution_time -start_execution_time)

        if(batch_count == 624):
            print(f"Total Time Taken for all Batches in Third Stage : {(total_time)*1000:.3f} ms")
            print(f"End Time: {end_execution_time * 1000:.3f} ms")
            print(f"Total Exceution Time: f{total_execution_time * 1000 :.3f} ms")
            break


def with_fuson_without_tensorrt():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = torchvision.models.GoogLeNet()
    
    stage_1 = get_third_stage_googleLeNet(model)

    # Fuse the model
    stage_1 = fuse_resnet(stage_1)

    stage_1.load_state_dict(torch.load(PATH_RESNET_LAYER_NO_TENSORRT_MODEL_2, weights_only = True))
    stage_1 = stage_1.to(device)
    total_time = 0
    total_execution_time = 0

    print("Waiting For PC1 to send activations")
    while True:
        data = socket.recv() 
        
        start_execution_time = time.time()

        activations = pickle.loads(data)
        activations = activations.to(device) 

        batch_start_time = time.time() 
        final_output = stage_1(activations) 
        batch_end_time = time.time()

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        print(f"Batch {batch_count} : Total Time: {(batch_end_time - batch_start_time) * 1000:.3f}")

        end_execution_time = time.time()
        total_execution_time += (end_execution_time - start_execution_time)
    
        if(batch_count == 624):
            print(f"Total Time Taken for {batch_count} batches in Third Stage: {total_time*1000:.2f} ms")
            print(f"End Time: {end_execution_time * 1000:.3f} ms")
            print(f"Total Execution Time: f{total_execution_time * 1000:.3f} ms")
            break

def with_fusion_with_tensorrt():
    batch_count = 0
    outputs = [] 

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_RESNET_LAYER_TENSORRT_ENGINE_2, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    total_time = 0
    total_execution_time=0
    batch_count = 0

    while True:
        data = socket.recv() 

        start_execution_time = time.time()
        activations = pickle.loads(data)

        # activations = torch.rand(16, 512, 4, 4)
        batch_start_time = time.time() 
        final_output = engine_infer_single_batch(engine, context, activations, batch_count=16)
        batch_end_time = time.time()         

        total_time += (batch_end_time - batch_start_time)
        batch_count += 1
        outputs.append(final_output)
        print(f"Batch {batch_count} | Total Time Taken: {(batch_end_time - batch_start_time)*1000:.3f} ms" )

        end_execution_time = time.time()
        total_execution_time += (end_execution_time - start_execution_time)

        if(batch_count == 624):
            print(f"Total Time Taken for all Batches in stage 3: {(total_time)*1000:.3f} ms")
            print(f"End Time: {end_execution_time * 1000:.3f} ms")
            print(f"Total Time Taken for {batch_count} batches in Third Stage: {total_time*1000:.2f} ms")
            break

def check_third():
    batch_count = 0
    total_time = 0
    outputs = []

    # Define the model
    model = torchvision.models.GoogLeNet()
    
    stage_0 = get_first_stage_googleLeNet(model)
    stage_1 = get_second_stage_googleLeNet(model)
    stage_2 = get_third_stage_googleLeNet(model)

    stage_2.load_state_dict(torch.load(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_2, weights_only = True))

    stage_0 = stage_0.to(device)
    stage_1 = stage_1.to(device)
    stage_2 = stage_2.to(device)
    

    total_time = 0
    data_loader = get_test_loader(16)

    for (i, data) in enumerate(data_loader):
        # torch.cuda.empty_cache()
        
        # input_data, label = data
        input_data = torch.randn(16, 3, 32, 32)
        input_data = input_data.to(device)

        stage_0_out = stage_0(input_data)
        stage_1_out = stage_1(stage_0_out)

        stage_1_out = stage_1_out.to(device)
        
        batch_start_time = time.time()  
        final_output = stage_2(stage_1_out)  
        batch_end_time = time.time()    

        final_output = final_output.to('cpu')  
        

        # outputs.append(final_output) Do not store the activations 
        
        total_time += (batch_end_time - batch_start_time)
        batch_count += 1

        print(f"Batch {batch_count}: arrived at PC3. Total Time Taken : {(batch_end_time - batch_start_time) * 1000 :.2f} ms")

        # Temperorary
        if(batch_count == 624):
            print(f"Total Tim Taken : {total_time*1000:.3f} ms")
            break

# check_third()

without_fusion_without_tensorrt()
# without_fusion_with_tensorrt()
# with_fuson_without_tensorrt()
# with_fusion_with_tensorrt()