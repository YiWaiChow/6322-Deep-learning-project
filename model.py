import torch
from torchvision import transforms
import torch.nn as nn

class Patch_Embedding(nn.Module):
    def __init__(self, channel, embed_dim, height, width, patch_size):
        super().__init__()
        self.in_dim = channel
        self.out_dim = embed_dim
        self.H = height
        self.W = width
        self.P = patch_size

        # if we are using convolution to replace patching, we may not need position embedding
        self.linear = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=self.P, stride=self.P, bias=True)
        self.norm = nn.LayerNorm([self.H/self.P, self.W/self.P, self.out_dim])

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SRA(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x   

class Stage_Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class PVT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


### I think we should be declaring the modules as class instead of function, kept the following as backup

# def Patch_Encoding_layer():
#     pass


# def Normalization_layer():
#     pass


# def Encoder_layer():
#     pass

# # modified MHA attention


# def SRA_layer():
#     pass


# def Feed_forward_layer():
#     pass


# def Spacial_Reduction_layer():
#     pass

# # multi head attention


# def MHA_layer():
#     pass


# def Encoder_layer():
#     pass


# # def PvT():
#     pass

