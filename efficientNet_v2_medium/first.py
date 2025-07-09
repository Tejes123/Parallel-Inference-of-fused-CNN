# ------------ common.py (imported by both PCs) ------------
import torch
import torchvision
import time
import tensorrt as trt
import zmq, pickle, time
import torch

from fuse_models import fuse_resnet
from data_loader import get_test_loader
from model_partition import get_first_stage_efficientNet
from onnx_file import engine_infer_single_batch

# Model without layer fuion without tensorrt 
PATH_NO_LAYER_NO_TENSORRT_MODEL_0 = r"stage_0/efficientNet_no_layer_no_tensorrt_0.pth"

# Model without fusion and with tensorrt
PATH_NO_LAYER_TENSORRT_ENGINE_0 = r"stage_0/efficientNet_no_layer_tensorrt_0.trt"

# Model with Fusion and witout Tensorrt 
PATH_LAYER_NO_TENSORRT_0 = r"stage_0/efficientNet_layer_no_tensorrt_0.pth"

# Model with Layer and with TensorRT
PATH_LAYER_TENSORRT_ONNX_0 = r"stage_0/efficientNet_layer_tensorrt_0.onnx"
PATH_LAYER_TENSORRT_ENGINE_0 = r"stage_0/efficientNet_layer_tensorrt_0.trt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://127.0.0.1:25900")     # PC0 pushes on port 5555
data_loader = get_test_loader(batch_size = 16)


def without_fusion_without_tensorrt():

    model = torchvision.models.efficientnet_v2_m()
    
    part_0 = get_first_stage_efficientNet(model)

    part_0.load_state_dict(torch.load(PATH_NO_LAYER_NO_TENSORRT_MODEL_0, weights_only=True))  
    stage_0 = part_0.to(device)

    

    socket.setsockopt(zmq.SNDBUF, 2* 1024)
    socket.setsockopt(zmq.SNDBUF, 2 * 1024 )

    print(int(socket.getsockopt(zmq.SNDBUF)))

    total_inference_time_time = 0

    start_inference = time.time()
    for (i, batch) in enumerate(data_loader):
        
        (input_data, label) = batch
        input_data = input_data.to(device)

        start_time = time.time()
        output_activation = stage_0(input_data)
        batch_time = time.time() - start_time

        # output_activation = output_activation.to("cpu")
        serialized_act = pickle.dumps(output_activation)
        
        socket.send(serialized_act)
        
        print(f"Batch {i} : Sent | Time Taken: {batch_time * 1000:.3f} ms")
        total_inference_time_time += batch_time
    end_inference = time.time()
    print(f"Total Inferene Time Taken: {total_inference_time_time * 1000:.3f} ms")
    print(f"Inference Start Time: {start_inference * 1000:.3f} ms")
    print(f"Total Execution time: {(end_inference - start_inference) * 1000:.3f} ms")
    socket.close()


def without_fusion_with_tensorrt():

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_NO_LAYER_TENSORRT_ENGINE_0, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    total_time = 0
    start_inference = time.time()
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
    end_inference = time.time()
    print(f"Total time taken: {total_time * 1000:.3f} ms")
    print(f"Inference Start Time: {start_inference * 1000:.3f} ms")
    print(f"Total Execution time: {(end_inference - start_inference) * 1000} ms")
    
def with_fusion_without_tensorrt():
    model = torchvision.models.efficientnet_v2_m()

    part_0 = get_first_stage_efficientNet(model)

    # Fuse the stage 0
    fused_part_0 = fuse_resnet(part_0)

    fused_part_0.load_state_dict(torch.load(PATH_LAYER_NO_TENSORRT_0, weights_only=True))  
    fused_part_0 = fused_part_0.to(device)

    total_time = 0
    start_inference = time.time()
    for (i, batch) in enumerate(data_loader):
        
        (input_data, label) = batch
        input_data = input_data.to(device)

        start_time = time.time()
        output_activation = fused_part_0(input_data)
        batch_time = time.time() - start_time

        serialized_act = pickle.dumps(output_activation)
        socket.send(serialized_act)
        

        print(f"Batch {i} : Sent | Time Taken: {batch_time * 1000:.3f} ms")
        total_time += batch_time
        
    end_inference = time.time()
    print(f"Total Time Taken: {total_time * 1000:.3f} ms")
    print(f"Inference Start Time: {start_inference * 1000:.3f} ms")
    print(f"Total Execution time: {(end_inference - start_inference) * 1000} ms")

def with_fusion_with_tensorrt():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(PATH_LAYER_TENSORRT_ENGINE_0, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    total_time = 0
    start_inference = time.time()
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
    end_inference = time.time()
    print(f"Total time taken: {total_time * 1000:.3f} ms")    
    print(f"Inference Start Time: {start_inference * 1000:.3f} ms")
    print(f"Total Execution time: {(end_inference - start_inference) * 1000} ms")

# without_fusion_without_tensorrt()  
# without_fusion_with_tensorrt() 
# with_fusion_without_tensorrt() 
with_fusion_with_tensorrt() 