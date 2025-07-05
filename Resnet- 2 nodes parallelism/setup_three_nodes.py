import torch
import torchvision
import os
from torch import nn
from fuse_models import fuse_resnet
from model_partition import get_first_stage_resnet, get_second_stage_resnet
from onnx_file import create_onnx_from_model, create_engine

"""
STAGE 0 PATHS
"""
# Model without layer fuion without tensorrt 
PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_0 = r"resnet_no_layer_no_tensorrt_0.pth"

# Model without fusion and with tensorrt
PATH_RESNET_NO_LAYER_TENSORRT_ONNX_0 = r"resnet_no_layer_tensorrt_0.onnx"
PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_0 = r"resnet_no_layer_tensorrt_0.trt"

# Model with Fusion and witout Tensorrt 
PATH_RESNET_LAYER_NO_TENSORRT_0 = r"resnet_layer_no_tensorrt_0.pth"

# Model with Layer and with TensorRT
PATH_RESNET_LAYER_TENSORRT_ONNX_0 = r"resnet_layer_tensorrt_0.onnx"
PATH_RESNET_LAYER_TENSORRT_ENGINE_0 = r"resnet_layer_tensorrt_0.trt"

# ---------------------------------------------------------------------------------WAIT

"""
STAGE 1 PATHS
"""

# Model without layer fuion without tensorrt 
PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_ONE = r"resnet_no_layer_no_tensorrt_1.pth"

# Model without fusion and with tensorrt
PATH_RESNET_NO_LAYER_TENSORRT_ONNX_ONE = r"resnet_no_layer_tensorrt_1.onnx"    
PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_ONE = r"resnet_no_layer_tensorrt_1.trt"

# Model with Fusion and witout Tensorrt 
PATH_RESNET_LAYER_NO_TENSORRT_MODEL_ONE = r"resnet_layer_no_tensorrt_1.pth"

# Model with Layer and with TensorRT
PATH_RESNET_LAYER_TENSORRT_ONNX_ONE = r"resnet_layer_tensorrt_1.onnx"
PATH_RESNET_LAYER_TENSORRT_ENGINE_ONE = r"resnet_layer_tensorrt_1.trt"

# batch_size = 16 

"""
sTEPS TO SET
1. Model without layer fuion without tensorrt - Done
2. Model without layer fusion with tensorrt - Done
3. Model with layer fusion and without tensorrt - 
4. Model eith layer fusion and without tensorrt - Done
"""

# a Function that returns the modified Resnet50 model for CIFAR 10 dataset - last layer has size 10 
def get_resnet50_for_CIFAR10():
    # CIFAR 10 has 10 classes, so the last layer should be a fully conneted with size 10
    net = torchvision.models.resnet50()

    # Get the number of input features to the last fully connected layer
    in_features = net.fc.in_features

    # Replace the last fully connected layer with a new one for 10 classes
    net.fc = nn.Linear(in_features, 10)
    return net

"""
STAGE 0 DEFINE ALL FUNCTIONS
"""

def save_model_witout_fusion_witout_tensorrt(stage):
    model = get_resnet50_for_CIFAR10()
    """
    Optional Training Script to train and save the model only Here
    """
    model_in_stage = None
    if(stage == 0):
        model_in_stage = get_first_stage_resnet(model)
        torch.save(model_in_stage.state_dict(), PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_0)
    else:
        model_in_stage = get_second_stage_resnet(model)
        torch.save(model_in_stage.state_dict(), PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_ONE)
    print(f"Model for stage {stage} saved ")
    

def save_model_without_fusion_with_tensorrt(stage):
    # First save the model
    if(stage == 0):
        if(os.path.exists(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_0)):
            # Model is not saved, so save it
            save_model_witout_fusion_witout_tensorrt(stage = 0)
        create_onnx_and_engine_for_stage0(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_0, fused = False)

    if(stage == 1):
        if(not(os.path.exists(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_ONE))):
            # Model is not saved, so save it
            save_model_witout_fusion_witout_tensorrt(stage = 1)
        
        create_onnx_and_engine_for_stage1(PATH_RESNET_NO_LAYER_NO_TENSORRT_MODEL_ONE, fused=False)


def save_model_with_fusion_without_tensorrt(stage):
    model = get_resnet50_for_CIFAR10()
    """
    If training, load the trained model
    """
    if(stage == 0):
        stage_0 = get_first_stage_resnet(model)
        fused_model = fuse_resnet(stage_0)
        torch.save(fused_model.state_dict(), PATH_RESNET_LAYER_NO_TENSORRT_0)
    elif stage == 1:
        stage_1 = get_second_stage_resnet(model)
        fused_model = fuse_resnet(stage_1)
        torch.save(fused_model.state_dict(), PATH_RESNET_LAYER_NO_TENSORRT_MODEL_ONE)

def save_model_with_fusion_with_tensorrt(stage):
    # First see if model is saved
    if(stage == 0):
        if(os.path.exists(PATH_RESNET_LAYER_NO_TENSORRT_0)):
            # Model is not saved, so save it
            save_model_with_fusion_without_tensorrt(stage = 0)
        create_onnx_and_engine_for_stage0(PATH_RESNET_LAYER_NO_TENSORRT_0, fused = True)

    if(stage == 1):
        if(not(os.path.exists(PATH_RESNET_LAYER_NO_TENSORRT_MODEL_ONE))):
            # Model is not saved, so save it
            save_model_with_fusion_without_tensorrt(stage = 1)
        
        create_onnx_and_engine_for_stage1(PATH_RESNET_LAYER_NO_TENSORRT_MODEL_ONE, fused=True)


def create_onnx_and_engine_for_stage0(model_path_stage_0, fused = False, batch_size = 16):
    model = get_resnet50_for_CIFAR10()
    stage0_resnet = get_first_stage_resnet(model)

    if(fused):
        stage0_resnet = fuse_resnet(stage0_resnet)
    # Load the model
    stage0_resnet.load_state_dict(torch.load(model_path_stage_0, weights_only=True))

    # Split the model
    stage0_resnet = get_first_stage_resnet(model)
    input_stage0_tensor = torch.rand(batch_size, 3, 32, 32)

    # Set the path 
    onnx_file_name_to_save = PATH_RESNET_NO_LAYER_TENSORRT_ONNX_0
    engine_file_name_to_save = PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_0

    # Change the path if fusion enabled
    if(fused):
        onnx_file_name_to_save = PATH_RESNET_LAYER_TENSORRT_ONNX_0
        engine_file_name_to_save = PATH_RESNET_LAYER_TENSORRT_ENGINE_0

    #Create ONNX file for split model
    create_onnx_from_model(stage0_resnet, onnx_file_name_to_save, input_stage0_tensor, batch_size) 

    # Crete the enhine for stage 0 ONNX file. This has dynamic input tensor shape
    min_input_shape = (16, 3, 32, 32)
    opt_input_shape = (16, 3, 224, 224)
    max_input_shape = (16, 3, 512, 512)
    create_engine(onnx_file_name_to_save, engine_file_name_to_save, min_input_shape, opt_input_shape, max_input_shape, batch_size=batch_size)


"""
STAGE 1
"""


def create_onnx_and_engine_for_stage1(model_path_stage_1, fused = False, batch_size = 16):
    model = get_resnet50_for_CIFAR10()

    # Split the model
    stage0_resnet = get_first_stage_resnet(model)
    stage1_resnet = get_second_stage_resnet(model)

    if(fused):
        stage1_resnet = fuse_resnet(stage1_resnet)

    # Load the model FOR STAGE 1
    stage1_resnet.load_state_dict(torch.load(model_path_stage_1, weights_only=True)) 

    # Get a sample output from previous stage. Use this as the tensir shape for creating the ONNX file
    """
    The out put of first stage is : (16, 512, 4, 4)
    """
    input_tensor = torch.rand(batch_size, 3, 32, 32)  
    stage0_output = stage0_resnet(input_tensor)
    stage0_output_shape = stage0_output.shape

    # SEt the path
        # Set the path 
    onnx_file_name = PATH_RESNET_NO_LAYER_TENSORRT_ONNX_ONE
    engine_file_name = PATH_RESNET_NO_LAYER_TENSORRT_ENGINE_ONE

    # Change the path if fusion enabled
    if(fused):
        onnx_file_name = PATH_RESNET_LAYER_TENSORRT_ONNX_ONE
        engine_file_name = PATH_RESNET_LAYER_TENSORRT_ENGINE_ONE

    # Create ONNX file for split model
    create_onnx_from_model(stage1_resnet, onnx_file_name,  stage0_output, batch_size) 

    # Create the Engine for stage 1 ONNX file . This has static input shape, so min, optimal and max input shape is same 
    engine_input_shape = tuple(stage0_output_shape)
    print(engine_input_shape)
    create_engine(onnx_file_name, engine_file_name, engine_input_shape, engine_input_shape, engine_input_shape, batch_size)

if __name__ == "__main__":
    # save_model_witout_fusion_witout_tensorrt(stage = 0)
    # save_model_witout_fusion_witout_tensorrt(stage = 1)

    # save_model_without_fusion_with_tensorrt(stage = 0)
    # save_model_without_fusion_with_tensorrt(stage = 1) 

    # save_model_with_fusion_without_tensorrt(stage = 0)
    # save_model_with_fusion_without_tensorrt(stage = 1)

    # save_model_with_fusion_with_tensorrt(stage = 0)
    save_model_with_fusion_with_tensorrt(stage = 1)