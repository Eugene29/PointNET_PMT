import torch
import tensorflow as tf
import numpy as np

## .py imports ##
from tf_PointNet import *
from read_point_cloud import * 
from utils import *

# clean_nohup_out()

def preprocess(fpath):
    ## Load/Preprocess Data
    print("preprocessed data not found. Preprecessing...")
    n_hits = get_pmtxyz("data/pmt_xyz.dat").size(0)
    X, y = torch.load(fpath, map_location=torch.device("cpu"))
    X = X.float() ## double to single float

    print("so far so good")
    new_X, F_dim, = preprocess_features(X, n_hits=n_hits) ## Output: [B, F, N] or [B, N, F]; F: (x, y, z, time, charge)

    ## Shuffle Data (w/ Seed)
    torch.save((new_X, y), "data/preprocessed_data.pt")
    print("Done!")