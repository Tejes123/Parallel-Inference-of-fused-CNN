import torch
import torchvision
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import time
import pycuda.autoinit # Very much important 
import torch.nn as nn 

from model_partition import get_first_stage_densenet, get_second_stage_densenet, get_third_stage_densenet


def create_onnx_from_model(model , onnx_path,  input_tensor = None, batch_size = 16):
    output_onnx="resnet50_fused_new.onnx"
    model.eval()
    
    # batch_size = 16
    # Generate input tensor with random values
    if(input_tensor == None):
        input_tensor = torch.rand(batch_size, 3, 32, 32)
    
    # Export torch model to ONNX
    print("[ONNX] Exporting ONNX model {}".format(onnx_path))
    torch.onnx.export(model, input_tensor, onnx_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"}},
        verbose=False)
    print(f"[ONNX] Exported to: {onnx_path}")
    

def create_engine(onnx_path, trtEngineName, min_input_shape, optimal_input_shape, max_input_shape, batch_size = 16, dynamic_input = True):
    # trtEngineName = "resnet50_fused_new.trt 

    TRT_LOGGER = trt.Logger()
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print(f"[ONNX] [ERROR]: Failed to parse ONNX model {onnx_path}")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)
    print(f"[ONNX_parse] ONNX file {onnx_path} parsed successfully")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 29)

    # Enable FP16 if supported
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    # The engine has dynamic input shape allocation - we specify the minimum ans maximum shape.
    profile.set_shape(input_name, min=min_input_shape,   # smallest batch you’ll ever send
                    opt= optimal_input_shape,   # “typical” size for fastest kernels       
                    max= max_input_shape )  # largest you want to handle)      

    config.add_optimization_profile(profile)

    print(f"[TRT_ENG] Building engine for: {onnx_path}")
    engine = builder.build_engine_with_config(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")

    with open(trtEngineName, "wb") as f:
        f.write(engine.serialize())

    print("[TRT_me] Engine serialized to " + trtEngineName)


def engine_infer_single_batch(engine, context, input_image, batch_count = 0):
    start_time = time.time()

    if(type(input_image) != type(np.ndarray(list()))):
        input_image = input_image.numpy()
    # print(type(input_image))
    # input_image = input_data
    

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_image.shape)

    # Allocate memory
    input_image = np.ascontiguousarray(input_image.astype(np.float32))
    output_shape = context.get_tensor_shape(output_name)
    output_size = int(np.prod(output_shape))
    # print(output_shape, output_size)
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    device_input = cuda.mem_alloc(input_image.nbytes)
    device_output = cuda.mem_alloc(output_size * np.dtype(output_dtype).itemsize)

    context.set_tensor_address(input_name, int(device_input))
    context.set_tensor_address(output_name, int(device_output))

    # Prepare output buffer
    host_output = np.empty(output_shape, dtype=output_dtype)

    stream = cuda.Stream()

    cuda.memcpy_htod_async(device_input, input_image, stream)
    # end_h2d.record(stream)

    # Inference
    # start_exec.record(stream)
    context.execute_async_v3(stream.handle)
    # end_exec.record(stream)

    # Device to Host
    # start_d2h.record(stream)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # end_d2h.record(stream)

    # Synchronize
    stream.synchronize()
    end_time = time.time()

    total_batch_time = (end_time - start_time) * 1000
    # total_time += batch_time
    # print(batch_time)
    # outputs.append(host_output[0])
    # print(f"Batch {batch_count} : Total Time:  {total_batch_time :.3f} ms")

    return host_output    

# create_onnx_from_model(model1, "sample.onnx")     