import torch
from torchvision import transforms
import torch.nn as nn

class Patch_Embedding(nn.Module):
    def __init__(self, channel, embed_dim, height, width):
        super().__init__()
        self.in_dim = channel       #input dimension/channel
        self.out_dim = embed_dim    #output dimension/embedding, we can try C1=32, C2=64, C3=128, C4=256
        self.H = height
        self.W = width
        self.P = 4

        # if we are using convolution to replace patching, we may not need position embedding
        self.linear = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=self.P, stride=self.P, bias=True)
        self.norm = nn.LayerNorm([self.H/self.P, self.W/self.P, self.out_dim])

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class SRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.q = nn.Linear()
        self.k = nn.Linear()
        self.v = nn.Linear()
        #TBC

    def forward(self, x):
        return x
    
class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #layer norm
        #SRA
        #result + residual
        #layer norm
        #feed forward
        #result + residual
        #reshape

    def forward(self, x):
        return x   

class Stage_Module(nn.Module):
    def __init__(self):
        super().__init__()
        #patch embedding
        #transformer encoder

    def forward(self, x):
        return x

class PVT(nn.Module):
    def __init__(self):
        super().__init__()
        #stage module 1
        #stage module 2
        #stage module 3
        #stage module 4

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

