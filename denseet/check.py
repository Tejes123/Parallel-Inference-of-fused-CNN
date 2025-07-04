import torchvision
import torch 

densenet = torchvision.models.densenet169()

x = torch.randn(16, 3, 32, 32)
out = densenet(x)

print("OutPut: ", out.shape)