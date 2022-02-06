import torch
import torch.nn as nn 
from torchvision import models
from PIL import Image
from torchvision import transforms
from torchinfo import summary
import matplotlib.pyplot as plt 
import numpy as np


class new_model(nn.Module):
    def __init__(self, sequential_list, linear=None):
        super().__init__()
        self.sequential_list = sequential_list
        self.net = nn.Sequential(*self.sequential_list)
        self.linear = linear
    def forward(self, x):
        x = self.net(x)
        if self.linear != None:
            x = x.reshape(x.shape[0], -1)
            x = self.linear(x)
        return x


#function takes a pretrained resnet model and creates a split model at the given layer name

def split_model(layer_name):
    pretrained = models.resnet50(pretrained=True)
    children_list_first=[]
    children_list_second=[]
    split_flag = 0
#generating 2 lists for 2 halfs of the model
    for n,c in pretrained.named_children():
        if split_flag == 1:
            children_list_second.append(c)
            continue
        children_list_first.append(c)
        if n == layer_name:
            split_flag = 1
    linear = children_list_second.pop(-1)
    print(len(children_list_first))
#returning the generated models through new_model class which accepts the list of layers as a paremeter
    return (new_model(children_list_first), new_model(children_list_second, linear))
    

resnet_split = split_model("layer3")

torch.save(resnet_split[0],'resnet_split_1.pt')
torch.save(resnet_split[1],'resnet_split_2.pt')