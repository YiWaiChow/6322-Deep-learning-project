import torch
from torchvision import transforms
import torch.nn as nn


class Patch_Embedding(nn.Module):
    def __init__(self, channel, embed_dim, height, width, patch_dim):
        super().__init__()
        self.in_dim = channel       # input dimension/channel
        # output dimension/embedding, we can try C1=32, C2=64, C3=128, C4=256
        self.out_dim = embed_dim

        # is this 4 or 16?
        # # # Not sure if we can do it dynamically, I guess we can just do patch_size = 4 8 16 32
        self.P = patch_dim

        # if we are using convolution to replace patching, we may not need position embedding
        # this outputs a shape of Batch size, embedding dimension, H, W
        self.linear = nn.Conv2d(
            channel, embed_dim, kernel_size=self.P, stride=self.P, bias=True)
        # self.norm = nn.LayerNorm([height/self.P, width/self.P, embed_dim])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        print(x.shape, "size of x before patch embedding, B, C, H, W \n")
        # flatten it into 2d, so H and W collapse into number of patches, then we swap the shape
        # from [B, ED, H,W] -> [B, ED, number of patches] -> [B, number of patches, ED]
        # this is done to follow the convention of the paper, where the embedding dimension is the last dimension
        x = self.linear(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        # output shape should be [B, Number of patches, ED], where number of patches should be HW/4*2
        print(x.shape, "size of x after patch embedding, B, Patches, ED \n")
        return x


# Spatial-Reduction Attention
class SRAttention(nn.module):
    def __init__(self, num_heads, channels, height, width, reduction_ratio):
        self.num_heads = num_heads
        self.head_dimension = channels/self.num_heads
        # i am not sure how to change the size here
        # self.dim = channels*self.head_dimension

        # the Weight is Ci X d Head, so the input dimension should be c and the output should be d head
        self.L = nn.Linear(self.c,
                           self.head_dimension)
        self.c = channels
        self.sr = SR(height, width, channels,
                     reduction_ratio, self.c)
        #  Wo has size Ci X Ci, this is becasuse d head = Ci/Ni, after concatnating N Ci, the dimension becomes Ci.
        self.L2 = nn.Linear(self.c, self.c)

    def forward(self, query, key, value):
        SRA = None
        for i in range(self.num_heads):
            # HW x d_head
            qi = self.L(query)
            # HW/R^2 x d_head
            srk = self.L(self.sr(key))
            # HW/R^2 x d_head
            srv = self.L(self.sr(value))
            # attention at stage i
            # HW X d_head @ d_head X HW/R^2 @ HW/R^2 x d_head = > HW X d_head <--- the shape of the A_i
            Ai = (torch.softmax(qi@srk.T/(self.head_dimension**0.5)))@srv
            if(SRA is None):
                SRA = Ai
            else:
                SRA = torch.cat(SRA, Ai)
        # SRA after concatinating should be HW X D_head*Ni -> HW X Ci
        SRA = self.L2(SRA)
        print(SRA.shape, "size of x after SRA, should be HW, Ci \n ")
        return SRA


# Spatial Reduction
# SR(x) = Norm(Reshape(x,Ri)W^s)
class SR(nn.Module):
    def __init__(self, height, width, channels, reduction_ratio):
        super().__init__()
        self.H = height
        self.W = width
        self.C = channels
        self.R = reduction_ratio
        # after reshaping x into HW/R^2 X R^2C, it takes in R^2C and projects to Ci
        self.linear_projection = nn.Linear(self.R**2*self.C, self.C)
        # then re layer norm on the number of channels
        self.norm = nn.LayerNorm(self.C)

    def forward(self, x):
        # reduced the sptial scale of x
        # by reshaping the sequence into size HW/R^2 X R^2C at stage i

        reduced_x = torch.reshape(
            x, [self.H*self.W/(self.R**2), (self.R**2*self.C)])
        new_x = self.linear_projection(reduced_x)
        new_x = self.norm(new_x)
        # output should be of size HW/R^2 x CI
        print(new_x.shape, "size of x after SR, should be HW/R^2 x CI, CI \n")
        return new_x


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
    def __init__(self, height, width, channels, reduction_ratio):
        super().__init__()
        self.num_heads = 2
        self.norm1 = nn.LayerNorm([height, width, channels])
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.a = SRAttention(self.num_heads, channels,
                             height, width, reduction_ratio)
        self.norm2 = nn.LayerNorm([height, width, channels])
        self.ff = Feed_Forward()  # missing i, h, o

    def forward(self, x):
        n1 = self.norm1(x)
        # q = n1?
        q = self.q(n1)
        k = self.k(n1)
        v = self.v(n1)
        # idk if qkv are already linear transform or not, in the paper it looks like its hasn't do the transform yet
        # # # I think you are right, linear transform of qkv should be within the SRAttention module
        a = self.a(self.num_heads, q, k, v)
        x += a
        n2 = self.norm2(x)
        ff = self.ff(n2)
        x += ff
        # we may not need reshape
        return x


class Stage_Module(nn.Module):
    # added patch_dim
    def __init__(self, channels, embedding_dim, Height, Width, reduction_ratio, patch_dim):
        super().__init__()
        # # # patch embedding
        self.PE = Patch_Embedding(channels, embedding_dim, Height, Width, patch_dim)
        self.TE = Transformer_Encoder(Height, Width, channels, reduction_ratio)
        # transformer encoder

    def forward(self, x):
        x = self.PE(x)
        x = torch.reshape(x, [Height, Width, ])
        return x


class PVT(nn.Module):
    def __init__(self):
        super().__init__()
        # input at stage 1 is H X W X 3
        # stage module 1

        # stage module 2
        # stage module 3
        # stage module 4

    def forward(self, x):
        return x


if __name__ == "__main__":

    # I think we should be declaring the modules as class instead of function, kept the following as backup

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
