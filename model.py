import torch
import torchvision
from torchvision import transforms
import torch.nn as nn


class Patch_Embedding(nn.Module):
    def __init__(self, channel, embed_dim, patch_dim):
        super().__init__()
        self.in_dim = channel
        self.out_dim = embed_dim

        self.P = patch_dim

        # this outputs a shape of Batch size, embedding dimension, H, W
        self.linear = nn.Conv2d(
            channel, embed_dim, kernel_size=patch_dim, stride=patch_dim, bias=True)
        # self.norm = nn.LayerNorm([height/self.P, width/self.P, embed_dim])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # flatten it into 2d, so H and W collapse into number of patches, then we swap the shape
        # from [B, ED, H,W] -> [B, ED, number of patches] -> [B, number of patches, ED]
        # this is done to follow the convention of the paper, where the embedding dimension is the last dimension

        x = self.linear(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # output shape should be [B, Number of patches, ED], where number of patches should be HW/4*2

        return x


# Spatial-Reduction Attention
class SRAttention(nn.Module):
    def __init__(self, num_heads, channels, height, width, reduction_ratio, batch_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = channels//self.num_heads

        self.c = channels

        # the Weight is Ci X d Head, so the input dimension should be c and the output should be d head
        self.L = nn.Linear(self.c,
                           self.head_dimension)
        self.sr = SR(height, width, channels,
                     reduction_ratio, batch_size)
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
            Ai = (torch.softmax(qi@srk.transpose(1, 2) /
                                (self.head_dimension**0.5), dim=1))@srv
            if(SRA is None):
                SRA = Ai
            else:

                SRA = torch.cat((SRA, Ai), dim=2)

        # SRA after concatinating should be HW X D_head*Ni -> HW X Ci
        SRA = self.L2(SRA)

        return SRA


# Spatial Reduction
# SR(x) = Norm(Reshape(x,Ri)W^s)
class SR(nn.Module):
    def __init__(self, height, width, channels, reduction_ratio, batch_size):
        super().__init__()
        self.H = height
        self.W = width
        self.C = channels
        self.B = batch_size
        self.R = reduction_ratio
        # after reshaping x into HW/R^2 X R^2C, it takes in R^2C and projects to Ci
        self.linear_projection = nn.Linear(self.R**2*self.C, self.C)
        # then re layer norm on the number of channels
        self.norm = nn.LayerNorm(self.C)

    def forward(self, x):
        # reduced the sptial scale of x
        # by reshaping the sequence into size HW/R^2 X R^2C at stage i

        reduced_x = torch.reshape(
            x, [self.B, self.H*self.W//(self.R**2), (self.R**2*self.C)])
        new_x = self.linear_projection(reduced_x)
        new_x = self.norm(new_x)
        # output should be of size HW/R^2 x CI

        return new_x


class Feed_Forward(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU(inplace=False)
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class Transformer_Encoder(nn.Module):
    def __init__(self, height, width, channels, reduction_ratio, patch_dim, batch_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(channels)
        self.a = SRAttention(self.num_heads, channels,
                             height//patch_dim, width//patch_dim, reduction_ratio, batch_size)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = Feed_Forward(channels, channels//2, channels)

    def forward(self, x):
        n1 = self.norm1(x)
        a = self.a(n1, n1, n1)
        x = torch.add(x, a)
        n2 = self.norm2(x)
        ff = self.ff(n2)
        x = torch.add(x, ff)
        return x


class Stage_Module(nn.Module):
    def __init__(self, channels, embedding_dim, Height, Width, reduction_ratio, patch_dim, batch_size, num_heads):
        super().__init__()
        self.H = Height
        self.W = Width
        self.out_dim = embedding_dim
        self.P = patch_dim
        self.B = batch_size
        self.PE = Patch_Embedding(channels, embedding_dim, patch_dim)
        self.TE = Transformer_Encoder(
            Height, Width, embedding_dim, reduction_ratio, patch_dim, batch_size, num_heads)

    def forward(self, x):
        x = self.PE(x)
        x = self.TE(x)
        # # # reshape to H(i-1)/P x W(i-1)/P x ED as output
        x = torch.reshape(x, [self.B, self.H//self.P,
                              self.W//self.P, self.out_dim]).permute([0, 3, 1, 2])
        return x


class PVT(nn.Module):
    def __init__(self, channels, height, width, batch_size):
        super().__init__()
        # input at stage 1 is H X W X 3

        self.stg1 = Stage_Module(channels, 64, height,
                                 width, reduction_ratio=8, patch_dim=4, batch_size=batch_size, num_heads=1)
        
        self.stg2 = Stage_Module(
            64, 128, height//4, width//4, reduction_ratio=4, patch_dim=2, batch_size=batch_size, num_heads=2)
        
        self.stg3 = Stage_Module(
            128, 256, height//8, width//8, reduction_ratio=2, patch_dim=2, batch_size=batch_size, num_heads=4)
        
        self.stg4 = Stage_Module(256, 512, height//16,
                                 width//16, reduction_ratio=1, patch_dim=2, batch_size=batch_size, num_heads=8)
        

        self.head = nn.linear(512)

    def forward(self, x):

        x = self.stg1(x)

        x = self.stg2(x)

        x = self.stg3(x)

        x = self.stg4(x)

        return x


class classification_pvt(nn.Module):
    def __init__(self, channels, height, width, batch_size, num_classes):
        super().__init__()
        # input at stage 1 is H X W X 3

        self.output_H = height//32
        self.output_W = width//32

        self.stg1 = Stage_Module(channels, 64, height,
                                 width, reduction_ratio=8, patch_dim=4, batch_size=batch_size, num_heads=1)
        self.stg2 = Stage_Module(
            64, 128, height//4, width//4, reduction_ratio=4, patch_dim=2, batch_size=batch_size, num_heads=2)
        self.stg3 = Stage_Module(
            128, 256, height//8, width//8, reduction_ratio=2, patch_dim=2, batch_size=batch_size, num_heads=4)
        self.stg4 = Stage_Module(256, 512, height//16,
                                 width//16, reduction_ratio=1, patch_dim=2, batch_size=batch_size, num_heads=8)

        self.head = nn.Linear(self.output_H*self.output_W*512, 128)
        self.head2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.stg1(x)

        x = self.stg2(x)

        x = self.stg3(x)

        x = self.stg4(x).permute([0, 2, 3, 1])

        x = x.view(-1, self.output_H*self.output_W*512)
        x = self.head(x)
        x = self.relu(x)
        x = self.head2(x)
        return x