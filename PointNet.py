import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import Linear
from time import time

# class STNkd(nn.Module):
'''
    T-Net (k-dimensions)
    Here ONLY for reference but not used due to its negligible performance improvement.
'''
#     def __init__(self, channel=None, k=64, dim_reduce_factor=1):
#         super(STNkd, self).__init__()
#         self.k = k
#         self.conv1 = torch.nn.Conv1d(channel if channel is not None else k, int(64//dim_reduce_factor), 1)
#         self.conv2 = torch.nn.Conv1d(int(64//dim_reduce_factor), int(128//dim_reduce_factor), 1)
#         self.conv3 = torch.nn.Conv1d(int(128//dim_reduce_factor), int(1024//dim_reduce_factor), 1)
#         self.fc1 = nn.Linear(int(1024//dim_reduce_factor), int(512//dim_reduce_factor))
#         self.fc2 = nn.Linear(int(512//dim_reduce_factor), int(256//dim_reduce_factor))
#         self.fc3 = nn.Linear(int(256//dim_reduce_factor), k * k)
#         self.relu = nn.ReLU()
#         self.dim_reduce_factor = dim_reduce_factor

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0] ## Q. why [0]? ## max(keepdim=True) returns (tensor, indices)
#         x = x.view(-1, int(1024//self.dim_reduce_factor))

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         ## add identity matrix for numerical stability. 
#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
#             batchsize, 1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x
    
class PointNetfeat(nn.Module):
    '''
    PointNet (Encoder) Implementation
    1. STNKD (DISCARDED due to no improvement)
    2. 1x1 Conv1D layers (Literally a linear layer)
    3. Global Statistics (Mean shows superior performance than Min/Max)
    '''
    def __init__(self, dimensions, dim_reduce_factor, args):
        super(PointNetfeat, self).__init__()
        dr = args.enc_dropout
        self.conv2lin = args.conv2lin
        # self.sparse = args.sparse
        F1, F2, self.latent_dim = [64, int(128 / dim_reduce_factor), int(1024 / dim_reduce_factor)] ## hidden dimensions

        if self.conv2lin:
            self.conv1 = Linear(dimensions, F1)  # lose a dimension after coordinate transform
            self.conv2 = Linear(F1, F2)
            self.conv3 = Linear(F2, self.latent_dim)
        else:
            self.conv1 = torch.nn.Conv1d(dimensions, F1, 1)  # lose a dimension after coordinate transform
            self.conv2 = torch.nn.Conv1d(self.conv1.out_channels, F2, 1)
            self.conv3 = torch.nn.Conv1d(self.conv2.out_channels, self.latent_dim, 1)
        self.dr1 = nn.Dropout(dr)
        self.dr2 = nn.Dropout(dr)
    
    def stats(self, x):
        meani = torch.mean(x, dim=2 if not self.conv2lin else 1)
        return meani
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.dr1(x))
        x = F.relu(self.dr2(self.conv2(x)))
        x = self.conv3(x)
        global_stats = self.stats(x)
        return global_stats
    
class PointClassifier(torch.nn.Module):
    def __init__(self, n_hits, dim, dim_reduce_factor, out_dim, args):
        '''
        ## Main Model ## 
        :param n_hits: number of points per point cloud
        :param dim: total dimensions of data (3 spatial + time and/or charge)
        '''
        super(PointClassifier, self).__init__()
        dr = args.dec_dropout
        self.n_hits = n_hits
        self.encoder = PointNetfeat(dimensions=dim,
                                    dim_reduce_factor=dim_reduce_factor,
                                    args=args,)
        # self.latent = self.encoder.latent_dim ## dimension from enc to dec
        self.decoder = nn.Sequential(
            nn.Dropout(p=dr),
            nn.Linear(self.encoder.latent_dim, int(512/dim_reduce_factor)),
            nn.LeakyReLU(),
            nn.Linear(int(512/dim_reduce_factor), int(128/dim_reduce_factor)),
            nn.LeakyReLU(),
            nn.Linear(int(128/dim_reduce_factor), out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
