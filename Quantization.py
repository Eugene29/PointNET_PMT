import torch
import accelerate
from accelerate import Accelerator
import os
import argparse
from time import time
from optimum.quanto import qint8
from optimum.quanto.calibrate import Calibration
from optimum.quanto.quantize import quantize, freeze

## .py imports ##
from PointNet import *
from read_point_cloud import * 
from utils import *

strt = time()
clean_nohup_out()
parser = argparse.ArgumentParser()
## Hyperparameters
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--reduce_lr_wait', type=int, default=20)
parser.add_argument('--enc_dropout', type=float, default=0.2)
parser.add_argument('--dec_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
parser.add_argument('--load_ver', help="version of the model: {int or \"Quantized\"}")
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--energy_mult', type=float, default=1)
parser.add_argument('--xyz_mult', type=float, default=1)
parser.add_argument('--conv2lin', action="store_true")
parser.add_argument('--scale_energy_loss', type=float, default=1)
parser.add_argument('--QAT', type=int, default=0)

## Initiate accelerator (distributed training) + flush output to {ver}/train.txt
args = parser.parse_args()
ver = args.load_ver
accelerator = accelerate.Accelerator()
args.accelerator = accelerator
ddp_kwargs = accelerate.DistributedDataParallelKwargs()
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
accelerator.print(f"ver: {ver}")
# init_logfile("Quantization")
init_logfile(ver, quantize=True)
print(type(ver))
script_strt = time()

## Load/Preprocess Data
with accelerator.main_process_first():
    ## Load data
    accelerator.print("loading data...")
    pmtxyz = get_pmtxyz("data/pmt_xyz.dat", accelerator=accelerator)
    X, y = torch.load(f"./data/train_X_y_ver_all_xyz_energy.pt", map_location=torch.device("cpu"))
    X = X.float() ## double to single float

    ## Preprocess (time, charge, x, y, z) -> (x, y, z, time, charge) and zero out inactive sensors' x, y, z. 
    accelerator.print("preprocessing data...")
    new_X, F_dim,= preprocess_features(X, n_hits=pmtxyz.size(0), args=args) ## [B, F, N] or [B, N, F]; F: (x, y, z, time, charge)

    ## update vars
    args.batch_size = int(args.batch_size // accelerator.num_processes) ##  divide the batch size by the number of GPU used. 
    n_data = new_X.shape[0]
    args.n_hits = pmtxyz.shape[0] ## num sensors

    ## Shuffle Data (w/ Seed)
    train_loader, val_loader, test_loader = shuffle_data(new_X=new_X, y=y, n_data=n_data, args=args)

## Init model
model = PointClassifier(
                n_hits=args.n_hits, 
                dim=F_dim, 
                out_dim=y.size(-1),
                dim_reduce_factor=args.dim_reduce_factor,
                args = args,
                )
nparam = sum([p.numel() for p in model.parameters()])
accelerator.print(f"num. parameters: {nparam}")

## prepare variables to load state
model, train_loader, val_loader, test_loader = accelerator.prepare(model, train_loader, val_loader, test_loader)
model.eval()
accelerator.wait_for_everyone()
if args.load_ver is not None:
    model_dir = f"./{ver}/model_dir"
    accelerator.load_state(model_dir)

##### Quantization #####
quantized_dir = f"{ver}/quantized_dir/"
os.makedirs(quantized_dir) if not os.path.exists(quantized_dir) else None

## Test the performance and check the size of the unquantized model
model(next(iter(train_loader))[0]) ## Kernel Warm Up
pred_all_data(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, accelerator=accelerator)
print_model_state_size(model, quantized_dir+"model.pth")

# Quantize Model in int8
quantize(model, weights=qint8, activations=qint8)
with torch.no_grad():
    with Calibration(momentum=0.9):
        for batch in train_loader:
            X, y = batch
            model(X)

## Quantization Aware Training (more like finetuning)
if args.QAT:
    ## New Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = accelerator.prepare(optimizer)
    current_lr = optimizer.param_groups[0]['lr']
    print(current_lr)
    model.train()
    for epoch in range(args.QAT):
        train_loss = 0 
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X, y = batch
            out = model(X)

            ## loss
            xyz_loss = F.mse_loss(out[:, :-1], y[:, :-1])
            energy_loss = args.scale_energy_loss * F.l1_loss(out[:, -1], y[:, -1])
            loss = xyz_loss + energy_loss
            train_loss += loss.item()

            ## gradient
            loss.backward()
            optimizer.step()
        accelerator.print(train_loss / len(train_loader))

## Change the model weights into qint
freeze(model)
accelerator.print(model)

## Save model weights and compute performance and size. 
torch.save(model.state_dict(), quantized_dir+"/model.pth")
print("Time and Performance after quantizaing: \n")
print_model_state_size(model, quantized_dir+"/model.pth")
pred_all_data(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, accelerator=accelerator)