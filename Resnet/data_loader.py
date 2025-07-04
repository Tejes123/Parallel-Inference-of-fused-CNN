import torch, torchvision 
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
 

def get_test_loader(batch_size = 16):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10( root = "D:\AI ML\Pipeline", train=False,
                                       download=False, transform=transform)
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def get_complete_loader(batch_size = 16):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root = "D:\AI ML\Pipeline", train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10( root = "D:\AI ML\Pipeline", train=False,
                                       download=False, transform=transform)
    
    totalset = trainset + testset

    total_loader = DataLoader(totalset, batch_size, shuffle = False)

    return total_loader  

