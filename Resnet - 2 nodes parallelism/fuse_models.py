import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.quantization import fuse_modules

# Step 2: Apply Layer Fusion
def fuse_resnet(model):
    """
    Fuses Conv-BN layers in a ResNet model using torch.quantization.fuse_modules.
    This is generally more robust for standard architectures like ResNet.
    Modifies the model in-place.
    """
    model.eval() # Ensure model is in eval mode for fusion

    # List of layers to fuse for a typical ResNet block structure
    # For a BasicBlock (used in ResNet18/34), it's conv1, bn1 and conv2, bn2
    # For a Bottleneck block (used in ResNet50/101/152), it's conv1, bn1; conv2, bn2; conv3, bn3;
    # and optionally downsample.0, downsample.1
    
    # We need to iterate through the model's modules and apply fusion recursively.
    # The `torch.quantization.fuse_modules` function works on a list of module names within a parent.

    # Function to apply fusion within a block
    def apply_fusion_to_block(block):
        if isinstance(block, models.resnet.BasicBlock):
            torch.quantization.fuse_modules(block, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
        elif isinstance(block, models.resnet.Bottleneck):
            torch.quantization.fuse_modules(block, [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3']], inplace=True)
        elif isinstance(block, nn.Sequential):
            # For sequential blocks, we need to iterate and try to fuse within them
            # This handles the stem and other sequential parts
            # Example: ResNet's initial conv and bn
            if hasattr(block, '0') and isinstance(block[0], nn.Conv2d) and \
               hasattr(block, '1') and isinstance(block[1], nn.BatchNorm2d):
               torch.quantization.fuse_modules(block, [['0', '1']], inplace=True)
        # Handle downsample path if it exists and is a sequential with conv-bn
        if hasattr(block, 'downsample') and block.downsample is not None:
            if isinstance(block.downsample, nn.Sequential) and \
               len(block.downsample) == 2 and \
               isinstance(block.downsample[0], nn.Conv2d) and \
               isinstance(block.downsample[1], nn.BatchNorm2d):
                torch.quantization.fuse_modules(block.downsample, [['0', '1']], inplace=True)

    # Traverse the model and apply fusion
    for name, module in model.named_children():
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for sub_name, sub_module in module.named_children():
                apply_fusion_to_block(sub_module)
        elif isinstance(module, (models.resnet.BasicBlock, models.resnet.Bottleneck)):
            apply_fusion_to_block(module)
        elif name == 'conv1' and isinstance(module, nn.Conv2d): # Handle the stem conv1
            # The stem conv1 and bn1 are usually direct children of the model
            if hasattr(model, 'bn1') and isinstance(model.bn1, nn.BatchNorm2d):
                print(f"Fusing model.conv1 and model.bn1...")
                fused_stem = torch.nn.utils.fuse_conv_bn_eval(model.conv1, model.bn1)
                model.conv1 = fused_stem
                model.bn1 = nn.Identity() # Replace BN with Identity
    print("Fusion done")
    return model
            
    # print("ResNet Conv-BN fusion complete using torch.quantization.fuse_modules.")