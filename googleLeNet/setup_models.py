import torch
import torchvision
import os
from torch import nn
from fuse_models import fuse_resnet
from model_partition import get_first_stage_googleLeNet, get_second_stage_googleLeNet, get_third_stage_googleLeNet
from onnx_file import create_onnx_from_model, create_engine

if(not(os.path.exists("stage_0"))):
    os.makedirs("stage_0")

if(not(os.path.exists("stage_1"))):
    os.makedirs("stage_1")

if(not(os.path.exists("stage_2"))):
    os.makedirs("stage_2")


"""
STAGE 0 PATHS
"""

# Model without layer fuion without tensorrt 
PATH_NO_LAYER_NO_TENSORRT_MODEL_0 = r"stage_0/googlelenet_no_layer_no_tensorrt_0.pth"

# Model without fusion and with tensorrt
PATH_NO_LAYER_TENSORRT_ONNX_0 = r"stage_0/googlelenet_no_layer_tensorrt_0.onnx"
PATH_NO_LAYER_TENSORRT_ENGINE_0 = r"stage_0/googlelenet_no_layer_tensorrt_0.trt"

# Model with Fusion and witout Tensorrt 
PATH_LAYER_NO_TENSORRT_0 = r"stage_0/googlelenet_layer_no_tensorrt_0.pth"

# Model with Layer and with TensorRT
PATH_LAYER_TENSORRT_ONNX_0 = r"stage_0/googlelenet_layer_tensorrt_0.onnx"
PATH_LAYER_TENSORRT_ENGINE_0 = r"stage_0/googlelenet_layer_tensorrt_0.trt"

# ---------------------------------------------------------------------------------WAIT

"""
STAGE 1 PATHS
"""

# Model without layer fuion without tensorrt 
PATH_NO_LAYER_NO_TENSORRT_MODEL_1 = r"stage_1/googlelenet_no_layer_no_tensorrt_1.pth"

# Model without fusion and with tensorrt
PATH_NO_LAYER_TENSORRT_ONNX_1 = r"stage_1/googlelenet_no_layer_tensorrt_1.onnx"    
PATH_NO_LAYER_TENSORRT_ENGINE_1 = r"stage_1/googlelenet_no_layer_tensorrt_1.trt"

# Model with Fusion and witout Tensorrt 
PATH_LAYER_NO_TENSORRT_MODEL_1 = r"stage_1/googlelenet_layer_no_tensorrt_1.pth"

# Model with Layer and with TensorRT
PATH_LAYER_TENSORRT_ONNX_1 = r"stage_1/googlelenet_layer_tensorrt_1.onnx"
PATH_LAYER_TENSORRT_ENGINE_1 = r"stage_1/googlelenet_layer_tensorrt_1.trt"


"""
Stage 2
"""

# Model without layer fuion without tensorrt 
PATH_NO_LAYER_NO_TENSORRT_MODEL_2 = r"stage_2/googlelenet_no_layer_no_tensorrt_2.pth"

# Model without fusion and with tensorrt
PATH_NO_LAYER_TENSORRT_ONNX_2 = r"stage_2/googlelenet_no_layer_tensorrt_2.onnx"    
PATH_NO_LAYER_TENSORRT_ENGINE_2 = r"stage_2/googlelenet_no_layer_tensorrt_2.trt"

# Model with Fusion and witout Tensorrt 
PATH_LAYER_NO_TENSORRT_MODEL_2 = r"stage_2/googlelenet_layer_no_tensorrt_2.pth"

# Model with Layer and with TensorRT
PATH_LAYER_TENSORRT_ONNX_2 = r"stage_2/googlelenet_layer_tensorrt_2.onnx"
PATH_LAYER_TENSORRT_ENGINE_2 = r"stage_2/googlelenet_layer_tensorrt_2.trt"


batch_size = 16 

"""
sTEPS TO SET
1. Model without layer fuion without tensorrt - Done
2. Model without layer fusion with tensorrt - Done
3. Model with layer fusion and without tensorrt - 
4. Model eith layer fusion and without tensorrt - Done
"""

model = torchvision.models.GoogLeNet()

stage0 = get_first_stage_googleLeNet(model)
stage1 = get_second_stage_googleLeNet(model)
stage2 = get_third_stage_googleLeNet(model)


# """
# STAGE 0 DEFINE ALL FUNCTIONS
# """

def save_model_witout_fusion_witout_tensorrt(stage):
    
    """
    Optional Training Script to train and save the model only Here
    """

    model_in_stage = None
    if(stage == 0):
        model_in_stage = get_first_stage_googleLeNet(model)
        torch.save(model_in_stage.state_dict(), PATH_NO_LAYER_NO_TENSORRT_MODEL_0)
    elif(stage == 1):
        model_in_stage = get_second_stage_googleLeNet(model)
        torch.save(model_in_stage.state_dict(), PATH_NO_LAYER_NO_TENSORRT_MODEL_1)
    else:
        model_in_stage = get_third_stage_googleLeNet(model)
        torch.save(model_in_stage.state_dict(), PATH_NO_LAYER_NO_TENSORRT_MODEL_2)
    print(f"Model for stage {stage} saved ")
    

def save_model_without_fusion_with_tensorrt(stage):
    # First save the model
    if(stage == 0):
        print("\n--------- STAGE 0 -------\n")
        if(not(os.path.exists(PATH_NO_LAYER_NO_TENSORRT_MODEL_0))):
            # Model is not saved, so save it
            save_model_witout_fusion_witout_tensorrt(stage = 0)
        create_onnx_and_engine_for_stage0(PATH_NO_LAYER_NO_TENSORRT_MODEL_0, fused = False)

    if(stage == 1):
        print("\n--------- STAGE 1 -------\n")
        if(not(os.path.exists(PATH_NO_LAYER_NO_TENSORRT_MODEL_1))):
            # Model is not saved, so save it
            save_model_witout_fusion_witout_tensorrt(stage = 1)
        
        create_onnx_and_engine_for_stage1(PATH_NO_LAYER_NO_TENSORRT_MODEL_1, fused=False)

    if(stage == 2):
        print("\n--------- STAGE 2 -------\n")
        if(not(os.path.exists(PATH_NO_LAYER_NO_TENSORRT_MODEL_2))):
            # Model is not saved, so save it
            save_model_witout_fusion_witout_tensorrt(stage = 2)
        
        create_onnx_and_engine_for_stage2(PATH_NO_LAYER_NO_TENSORRT_MODEL_2, fused=False)


def save_model_with_fusion_without_tensorrt(stage):
    """
    If training, load the trained model
    """
    if(stage == 0):
        print("\n---------STAGE 0-------\n")
        stage_0 = get_first_stage_googleLeNet(model)
        fused_model = fuse_resnet(stage_0)
        torch.save(fused_model.state_dict(), PATH_LAYER_NO_TENSORRT_0)
    elif stage == 1:
        print("\n---------STAGE 1-------\n")
        stage_1 = get_second_stage_googleLeNet(model)
        fused_model = fuse_resnet(stage_1)
        torch.save(fused_model.state_dict(), PATH_LAYER_NO_TENSORRT_MODEL_1)

    elif stage == 2:
        print("\n---------STAGE 2-------\n")
        stage_2 = get_third_stage_googleLeNet(model)
        fused_model = fuse_resnet(stage_2)
        torch.save(fused_model.state_dict(), PATH_LAYER_NO_TENSORRT_MODEL_2)

def save_model_with_fusion_with_tensorrt(stage):
    # First see if model is saved
    if(stage == 0):
        print("\n--------- STAGE 0 -------\n")
        if(not(os.path.exists(PATH_LAYER_NO_TENSORRT_0))):
            # Model is not saved, so save it
            save_model_with_fusion_without_tensorrt(stage = 0)
        create_onnx_and_engine_for_stage0(PATH_LAYER_NO_TENSORRT_0, fused = True)

    if(stage == 1):
        print("\n--------- STAGE 1 -------\n")
        if(not(os.path.exists(PATH_LAYER_NO_TENSORRT_MODEL_1))):
            # Model is not saved, so save it
            save_model_with_fusion_without_tensorrt(stage = 1)
        
        create_onnx_and_engine_for_stage1(PATH_LAYER_NO_TENSORRT_MODEL_1, fused=True)

    if(stage == 2):
        print("\n--------- STAGE 2 -------\n")
        if(not(os.path.exists(PATH_LAYER_NO_TENSORRT_MODEL_2))):
            # Model is not saved, so save it
            save_model_with_fusion_without_tensorrt(stage = 2)
        
        create_onnx_and_engine_for_stage2(PATH_LAYER_NO_TENSORRT_MODEL_2, fused=True)



def create_onnx_and_engine_for_stage0(model_path_stage_0, fused = False, batch_size = 16):
    model = torchvision.models.GoogLeNet()

    stage0_resnet = get_first_stage_googleLeNet(model)

    if(fused):
        stage0_resnet = fuse_resnet(stage0_resnet)
    # Load the model
    stage0_resnet.load_state_dict(torch.load(model_path_stage_0, weights_only=True))

    input_stage0_tensor = torch.rand(batch_size, 3, 32, 32)

    # Set the path 
    onnx_file_name_to_save = PATH_NO_LAYER_TENSORRT_ONNX_0
    engine_file_name_to_save = PATH_NO_LAYER_TENSORRT_ENGINE_0

    # Change the path if fusion enabled
    if(fused):
        onnx_file_name_to_save = PATH_LAYER_TENSORRT_ONNX_0
        engine_file_name_to_save = PATH_LAYER_TENSORRT_ENGINE_0

    #Create ONNX file for split model
    create_onnx_from_model(stage0_resnet, onnx_file_name_to_save, input_stage0_tensor, batch_size) 

    # Crete the enhine for stage 0 ONNX file. This has dynamic input tensor shape
    min_input_shape = (16, 3, 32, 32)
    opt_input_shape = (16, 3, 224, 224)
    max_input_shape = (16, 3, 512, 512)
    create_engine(onnx_file_name_to_save, engine_file_name_to_save, min_input_shape, opt_input_shape, max_input_shape, batch_size=batch_size)


# """
# STAGE 1
# """


def create_onnx_and_engine_for_stage1(model_path_stage_1, fused = False, batch_size = 16):
    model = torchvision.models.GoogLeNet()
    

    # Split the model
    stage0_densenet = get_first_stage_googleLeNet(model)
    stage1_densenet = get_second_stage_googleLeNet(model)

    if(fused):
        stage1_densenet = fuse_resnet(stage1_densenet)

    # Load the model FOR STAGE 1
    stage1_densenet.load_state_dict(torch.load(model_path_stage_1, weights_only=True)) 

    # Get a sample output from previous stage. Use this as the tensir shape for creating the ONNX file
    """
    The out put of first stage is : 16, 1024, 2, 2 
    """
    input_tensor = torch.rand(batch_size, 3, 32, 32)  
    stage0_output = stage0_densenet(input_tensor)
    stage0_output_shape = stage0_output.shape

    # Set the path 
    onnx_file_name = PATH_NO_LAYER_TENSORRT_ONNX_1
    engine_file_name = PATH_NO_LAYER_TENSORRT_ENGINE_1

    # Change the path if fusion enabled
    if(fused):
        onnx_file_name = PATH_LAYER_TENSORRT_ONNX_1
        engine_file_name = PATH_LAYER_TENSORRT_ENGINE_1

    # Create ONNX file for split model
    create_onnx_from_model(stage1_densenet, onnx_file_name, stage0_output, batch_size) 

    # Create the Engine for stage 1 ONNX file . This has static input shape, so min, optimal and max input shape is same 
    engine_input_shape = tuple(stage0_output_shape)
    print("Shape 0 Output:    ", engine_input_shape)
    create_engine(onnx_file_name, engine_file_name, engine_input_shape, engine_input_shape, engine_input_shape, batch_size)


def create_onnx_and_engine_for_stage2(model_path_stage_2, fused = False, batch_size = 16):
    model = model = torchvision.models.GoogLeNet()

    # Split the model
    stage0_densenet = get_first_stage_googleLeNet(model)
    stage1_densenet = get_second_stage_googleLeNet(model)
    stage_2_densenet = get_third_stage_googleLeNet(model)

    if(fused):
        stage_2_densenet = fuse_resnet(stage_2_densenet)

    # Load the model FOR STAGE 1
    stage_2_densenet.load_state_dict(torch.load(model_path_stage_2, weights_only=True)) 

    # Get a sample output from previous stage. Use this as the tensir shape for creating the ONNX file
    """
    The out put of first second is : (16, 1024, 2, 2)
    """
    input_tensor = torch.rand(batch_size, 3, 32, 32)  

    stage0_output = stage0_densenet(input_tensor)
    stage1_output = stage1_densenet(stage0_output)
    stage1_output_shape = stage1_output.shape

    # SEt the path
        # Set the path 
    onnx_file_name = PATH_NO_LAYER_TENSORRT_ONNX_2
    engine_file_name = PATH_NO_LAYER_TENSORRT_ENGINE_2

    # Change the path if fusion enabled
    if(fused):
        onnx_file_name = PATH_LAYER_TENSORRT_ONNX_2
        engine_file_name = PATH_LAYER_TENSORRT_ENGINE_2

    # Create ONNX file for split model
    create_onnx_from_model(stage_2_densenet, onnx_file_name, stage1_output, batch_size) 

    # Create the Engine for stage 1 ONNX file . This has static input shape, so min, optimal and max input shape is same 
    engine_input_shape = tuple(stage1_output_shape)
    print("Shape 1 Output:  ", engine_input_shape)
    create_engine(onnx_file_name, engine_file_name, engine_input_shape, engine_input_shape, engine_input_shape, batch_size)


if __name__ == "__main__":
    print("\n\n @@@@@@@@@@@@@@@@@@@@ WITHOUT FUSION WITHOUT TENSORRT @@@@@@@@@@@@@@@\n\n")
    save_model_witout_fusion_witout_tensorrt(stage = 0)  
    save_model_witout_fusion_witout_tensorrt(stage = 1)
    save_model_witout_fusion_witout_tensorrt(stage = 2)

    print("\n\n @@@@@@@@@@@@@@@@@@@@ WITHOUT FUSION WITH TENSORRT @@@@@@@@@@@@@@@\n\n")
    # save_model_without_fusion_with_tensorrt(stage = 0)
    save_model_without_fusion_with_tensorrt(stage = 1)
    save_model_without_fusion_with_tensorrt(stage = 2)

    print("\n\n @@@@@@@@@@@@@@@@@@@@ WITH FUSION WITHOUT TENSORRT @@@@@@@@@@@@@@@\n\n")
    save_model_with_fusion_without_tensorrt(stage = 0)
    save_model_with_fusion_without_tensorrt(stage = 1)
    save_model_with_fusion_without_tensorrt(stage = 2)

    print("\n\n @@@@@@@@@@@@@@@@@@@@ WITH FUSION WITH TENSORRT @@@@@@@@@@@@@@@\n\n")
    save_model_with_fusion_with_tensorrt(stage = 0)
    save_model_with_fusion_with_tensorrt(stage = 1)
    save_model_with_fusion_with_tensorrt(stage = 2)  