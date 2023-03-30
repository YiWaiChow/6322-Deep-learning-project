import torch
from torchvision import transforms
import torch.nn as nn

class Patch_Embedding(nn.Module):
    def __init__(self, channel, embed_dim, height, width):
        super().__init__()
        self.in_dim = channel       # input dimension/channel
        self.out_dim = embed_dim    # output dimension/embedding, we can try C1=32, C2=64, C3=128, C4=256
        self.P = 4

        # if we are using convolution to replace patching, we may not need position embedding
        self.linear = nn.Conv2d(channel, embed_dim, kernel_size=self.P, stride=self.P, bias=True)
        self.norm = nn.LayerNorm([height/self.P, width/self.P, embed_dim])

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class Attention(nn.module):
    def __init__(self, num_heads, q, k, v):
        pass
    def forward(self, x):
        return x

class SRA(nn.Module):
    def __init__(self, k, v):
        super().__init__()
        pass
    
    def forward(self, x):
        return x
    
class Feed_Forward(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.num_heads = 2
        self.norm1 = nn.LayerNorm([height, width, channels])
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.sra = SRA()
        self.a = Attention(self.num_heads)
        self.norm2 = nn.LayerNorm([height, width, channels])
        self.ff = Feed_Forward() ### missing i, h, o

    def forward(self, x):
        n1 = self.norm1(x)
        q = self.q(n1)
        k = self.k(n1)
        v = self.v(n1)
        kv = self.sra(k, v)
        a = self.a(self.num_head, q, kv)
        x += a 
        n2 = self.norm2(x)
        ff = self.ff(n2)
        x += ff
        # we may not need reshape
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

