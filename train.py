import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
import os
import argparse
from tqdm import tqdm
from pprint import pprint
from time import time

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
parser.add_argument('--use_wandb', action="store_true")
parser.add_argument('--reduce_lr_wait', type=int, default=20)
parser.add_argument('--enc_dropout', type=float, default=0.2)
parser.add_argument('--dec_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--smaller_run', action="store_true")
parser.add_argument('--save_ver', default=0)
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--energy_mult', type=float, default=1)
parser.add_argument('--xyz_mult', type=float, default=1)
parser.add_argument('--conv2lin', action="store_true")
parser.add_argument('--scale_energy_loss', type=float, default=1)

## Initiate accelerator (distributed training) + flush output to {ver}/train.txt
args = parser.parse_args()
ver = args.save_ver
accelerator = accelerate.Accelerator()
os.environ["WANDB_DISABLED"] = str(not args.use_wandb or args.debug)
ddp_kwargs = accelerate.DistributedDataParallelKwargs()
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
accelerator.print(f"ver: {ver}")
init_logfile(ver)
script_strt = time()

## model save name
model_dir = f"./{ver}/model_dir/"

## Load/Preprocess Data
with accelerator.main_process_first():
    ## Load data
    accelerator.print("loading data...")
    pmtxyz = get_pmtxyz("data/pmt_xyz.dat", accelerator=accelerator)
    X, y = torch.load(f"./data/train_X_y_ver_all_xyz_energy.pt", map_location=torch.device("cpu"))
    X = X.float() ## double to single float
    if args.debug:
        accelerator.print("debug got called")
        small = 5000
        X, y = X[:small], y[:small]

    ## Preprocess (time, charge, x, y, z) -> (x, y, z, time, charge) and zero out inactive sensors' x, y, z. 
    accelerator.print("preprocessing data...")
    new_X, F_dim,= preprocess_features(X, n_hits=pmtxyz.size(0), args=args) ## Output: [B, F, N] or [B, N, F]; F: (x, y, z, time, charge)

    ## update vars
    args.batch_size = int(args.batch_size // accelerator.num_processes) ## Divide the batch size by the number of GPUs 
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

## Optimizers, Scheduler, Prepare Distributed Training
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, "min", 0.80, patience=args.patience*accelerator.num_processes, min_lr=1e-5, threshold=1e-3, threshold_mode='rel') 
model, train_loader, val_loader, test_loader, optimizer, scheduler = accelerator.prepare(model, train_loader, val_loader, test_loader, optimizer, scheduler)
args.model, args.nparam, args.n_device, args.accelerator = model, nparam, accelerator.num_processes, accelerator

## log and print model config
args.batch_size *= accelerator.num_processes ## bring the batchsize back up ONLY for logging purposes
accelerator.init_trackers("pointNET", config=vars(args))
if accelerator.is_main_process:
    print("\n\nHyperparameters being used:\n")
    pprint(vars(args))
    print("\n\n")

## extract args vars
epochs = range(args.epochs)
xyz_mult = args.xyz_mult ## scale xyz to control spread better
energy_mult = args.energy_mult ## scale energy loss
if accelerator.is_main_process:
    pbar = tqdm(total=args.epochs, mininterval=10)

## Train Loop
best_val, best_train = float("inf"), float("inf")
train_lst, val_time_lst = [], []
for epoch in epochs:
    model.train()
    train_loss, tot_xyz_loss, tot_energy_loss = 0, 0, 0
    for i, batch in enumerate(train_loader):
        ## forward prop
        X, y = batch
        out = model(X)
        xyz_loss = F.mse_loss(out[:, :-1], y[:, :-1])
        energy_loss = args.scale_energy_loss * F.l1_loss(out[:, -1], y[:, -1]) ## Use L1 loss to dampen loss from falsely predicting tail labels
        loss = xyz_loss + energy_loss

        ## backward prop & update grad
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        ## logging
        with torch.no_grad():
            tot_xyz_loss += xyz_loss.item()
            tot_energy_loss += energy_loss.item()
            train_loss += loss.item()

    ## scale loss
    train_loss /= len(train_loader)
    tot_xyz_loss /= len(train_loader)
    tot_energy_loss /= len(train_loader)
    train_lst.append(train_loss)
    scheduler.step(train_loss)

    ## print
    accelerator.print(f"xyz loss: {tot_xyz_loss:.2f}")
    accelerator.print(f"energy loss: {tot_energy_loss:.2f}")
    accelerator.print(f"min train loss: {min(train_lst)}")

    ## validation
    mod = 5
    if epoch % mod == 0 or epoch == args.epochs-1:
        val_strt = time()
        model.eval()
        ## predict
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                X, y = batch
                out = model(X)
                out, y = accelerator.gather(out), accelerator.gather(y)
                val_loss += F.mse_loss(out, y).item()

        ## logging and printing
        val_loss /= len(val_loader)
        val_time_lst.append(time() - val_strt)
        current_lr = optimizer.param_groups[0]['lr']
        accelerator.print(f'\nepoch {epoch}, train loss: {train_loss:.2f}, val loss: {val_loss:.2f}, lr: {current_lr}')
        accelerator.log({
            "train loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        ## Update lr Scheduler and tqdm 
        if accelerator.is_local_main_process: ## only show update once
            pbar.update(mod) 

        ## save best (val loss) model
        cond = val_loss < best_val and epoch > args.epochs / 2 ## can only save from 2nd half of training
        if cond:
            best_val = val_loss
            accelerator.print(f"New Best Score!! Best val loss: {best_val}")
            accelerator.save_state(output_dir=model_dir)
            
        ## val epoch time
        if len(val_time_lst) > 2:
            accelerator.print(f"average val epoch time: {sum(val_time_lst[2:]) / (len(val_time_lst) - 2):.4f}") ## First few are unstable.
    else:
        ## if no validation
        accelerator.log({"train loss": train_loss})


accelerator.wait_for_everyone()
accelerator.load_state(model_dir) ## to match train.txt and plot
accelerator.print(f"Model saved in {model_dir}")

## Test loop
with torch.no_grad():
    abs_diff, test_loss = [], 0
    model.eval()
    for batch in test_loader:
        X, y = batch
        out = model(X)
        out, y = accelerator.gather(out), accelerator.gather(y)
        test_loss += F.mse_loss(out, y).item()
        abs_diff.append((y - out).abs())
    test_loss /= len(test_loader)

    ## Logging
    abs_diff = torch.cat(abs_diff)
    abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = abs_diff.mean(dim=0).cpu().numpy()
    abs_dict = {"abs_x_diff": abs_x_diff, "abs_y_diff": abs_y_diff, "abs_z_diff": abs_z_diff, "abs_energy_diff": abs_energy_diff}

## logging
tot_time = time() - script_strt
accelerator.log({"time": tot_time})
accelerator.print(f"total time taken: {tot_time:.0f} secs")
avg_val_time = sum(val_time_lst) / len(val_time_lst)

## min values
min_train = round(min(train_lst), 2)
loss_n_time = {"min_train": min_train, "min_val": best_val, "test_loss": test_loss, "avg_val_time": avg_val_time}
log_dict = {**abs_dict, **loss_n_time} ## recall: ** is for unpacking keys: vals (kwargs for short)
if accelerator.is_local_main_process:
    # print("\n")
    pprint(log_dict)
accelerator.log(log_dict) 
accelerator.wait_for_everyone()