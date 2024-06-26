import torch
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
import numpy as np
from accelerate.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader


def init_logfile(i):
    '''
        create and set logfile to be written. Also write init messages such as args and seed
    '''
    save_dir = f"./{i}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = open(f"{i}/train.txt", 'w', buffering=1)
    sys.stderr = log_file
    sys.stdout = log_file
    return save_dir, log_file

def clean_nohup_out():
    with open("nohup.out", "w") as f:
        pass

def preprocess_features(X, n_hits, args):
    '''
        1. swap the syntax (time, charge, x, y, z) -> (x, y, z, time, charge)
        2. zero out xyz positions of sensors that are not activated
    '''
    new_X = X.clone() ## same as deepcopy's copy() but optimzied for tensors
    new_X = new_X[:, :, [2, 3, 4, 0, 1]]
    new_X = new_X.mT

    ## create label tensor that contains 0 (not activated) or 1 (activated)
    label_feat = ((new_X[:, 3, :] != 0) & (new_X[:, 4, :] != 0)).float() ## register 0 if both time and charge == 0.
    print(f"Training Data shape: {new_X.shape}")
    label = label_feat.view(-1, 1, n_hits) ## add dimension to allow broadcasting

    ## zero out sensors not activated (including the position features as well)
    new_X = new_X * label
    F_dim = new_X.size(-2)
    new_X = new_X.mT if args.conv2lin else new_X
    return new_X, F_dim

def load_model(model, state_dict):
    '''
        Manually load data. However, using accelerate's load_state is preferred, which also saves the rng state, etc. 
    '''
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model

def shuffle_data(new_X, y, n_data, args):
    '''
        shuffle data and return train, val, test loader
    '''
    ## shuffle idx
    np.random.seed(seed=args.seed)
    set_seed(seed=args.seed)
    idx = np.random.permutation(n_data)
    saved = new_X.clone(), y.clone()
    new_X, y = new_X[idx], y[idx]
    assert torch.ne(saved[0], new_X).any() and torch.ne(saved[1], y).any(), "shuffling failed"

    ## create tensordataset
    train_split = 0.6
    val_split = 0.2
    train_idx = int(n_data * train_split)
    val_idx = int(train_idx + n_data * val_split)
    train = TensorDataset(new_X[:train_idx], y[:train_idx])
    val = TensorDataset(new_X[train_idx:val_idx], y[train_idx:val_idx])
    test = TensorDataset(new_X[val_idx:], y[val_idx:])

    ## create Dataloader
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    return train_loader, val_loader, test_loader

def plot_reg(diff, dist, test_loss, abs_diff, save_name):
    '''
        Create and save a plot of 6 total histogram (1 fig). 
    '''
    abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = abs_diff.mean(dim=0)
    energy_diff = torch.cat(diff["energy"], dim=0).cpu()
    energy_pred = torch.cat(dist["energy_pred"], dim=0).cpu()
    energy = torch.cat(dist["energy"], dim=0).cpu()

    x_diff = torch.cat(diff["x"], dim=0).cpu()
    y_diff = torch.cat(diff["y"], dim=0).cpu()
    z_diff = torch.cat(diff["z"], dim=0).cpu()

    x_pred = torch.cat(dist["x_pred"], dim=0).cpu()
    y_pred = torch.cat(dist["y_pred"], dim=0).cpu()
    z_pred = torch.cat(dist["z_pred"], dim=0).cpu()

    x = torch.cat(dist["x"], dim=0).cpu()
    y = torch.cat(dist["y"], dim=0).cpu()
    z = torch.cat(dist["z"], dim=0).cpu()

    plt.close()
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    fig.suptitle(f"Test MSE: {test_loss:.2f} (MSE(x) + MSE(y) + MSE(y) + MSE(energy))\n\
    Avg. abs. diff. in x={abs_x_diff:.2f}, y={abs_y_diff:.2f}, z={abs_z_diff:.2f}, energy={abs_energy_diff:.2f}")

    ## diff. plots
    x_diff_range = (-50, 50)
    axes[0,0].hist(x_diff, bins=20, range=x_diff_range, edgecolor='black')
    axes[0,0].set_title(r"x_diff ($x - \hat{x}$)")
    axes[0,0].set_xlabel('x diff')
    axes[0,0].set_ylabel('freq')

    y_diff_range = (-50, 50)
    axes[0,1].hist(y_diff, bins=20, range=y_diff_range, edgecolor='black')
    axes[0,1].set_title(r"y_diff ($y - \hat{y}$)")
    axes[0,1].set_xlabel('y diff')
    axes[0,1].set_ylabel('freq')

    z_diff_range = (-50, 50)
    axes[0,2].hist(z_diff, bins=20, range=z_diff_range, edgecolor='black')
    axes[0,2].set_title(r"z_diff ($z - \hat{z}$)")
    axes[0,2].set_xlabel('z diff')
    axes[0,2].set_ylabel('freq')

    energy_diff_range = (-0.5, 0.5)
    axes[0,3].hist(energy_diff, bins=20, range=energy_diff_range, edgecolor='black')
    axes[0,3].set_title(r"energy_diff ($energy - \hat{energy}$)")
    axes[0,3].set_xlabel('energy diff')
    axes[0,3].set_ylabel('freq')

    ## dist. plots
    x_range = (-250, 250)
    axes[1,0].hist(x, bins=20, range=x_range, edgecolor='black', label="x")
    axes[1,0].hist(x_pred, bins=20, range=x_range, edgecolor='blue', label=r'$\hat{x}$', alpha=0.5)
    axes[1,0].set_title("x dist")
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('freq')

    y_range = (-250, 250)
    axes[1,1].hist(y, bins=20, range=y_range, edgecolor='black', label="y")
    axes[1,1].hist(y_pred, bins=20, range=y_range, edgecolor='blue', label=r'$\hat{y}$', alpha=0.5)
    axes[1,1].set_title("y dist")
    axes[1,1].set_xlabel('y')
    axes[1,1].set_ylabel('freq')

    z_range = (-250, 250)
    axes[1,2].hist(z, bins=20, range=z_range, edgecolor='black', label="z")
    axes[1,2].hist(z_pred, bins=20, range=z_range, edgecolor='blue', label=r'$\hat{z}$', alpha=0.5)
    axes[1,2].set_title("z dist")
    axes[1,2].set_xlabel(r'z')
    axes[1,2].set_ylabel('freq')

    energy_range = (-1, 4)
    # energy_range = (-10, 10)
    axes[1,3].hist(energy, bins=20, range=energy_range, edgecolor='black', label="label")
    axes[1,3].hist(energy_pred, bins=20, range=energy_range, edgecolor='blue', label="pred", alpha=0.5)
    axes[1,3].set_title("energy")
    axes[1,3].set_xlabel('energy')
    axes[1,3].set_ylabel('freq')

    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 2].legend()
    axes[1, 3].legend()

    plt.savefig(save_name + "_hist.png")
    plt.close()

def pred_all_data(model, train_loader, val_loader, test_loader, accelerator):
    '''
        Predict on all data (train, val, test), print each loss, and the time it took.
    '''
    torch.cuda.empty_cache()
    model.eval()
    strt = time()
    with torch.no_grad():
        train_loss = 0
        out_lst = []
        y_lst = []
        for batch in train_loader:
            X, y = batch
            out = model(X)
            out, y = accelerator.gather(out), accelerator.gather(y)
            train_loss += F.mse_loss(out, y).item()
            out_lst.append(out)
            y_lst.append(y)
        train_loss /= len(train_loader)

        ## val (confirm) loop
        val_loss = 0
        for batch in val_loader:
            X, y = batch
            out = model(X)
            out, y = accelerator.gather(out), accelerator.gather(y)
            val_loss += F.mse_loss(out, y).item()
        val_loss /= len(val_loader)

        ## test loop
        abs_diff, test_loss = [], 0
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

    ## time
    tot_time = time() - strt
    accelerator.print(f"\nTotal time taken: {tot_time:.2f} secs\n")

    ## recall: ** is for unpacking keys: vals (kwargs for short)
    log_dict = {**abs_dict, **{"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}}
    if accelerator.is_local_main_process:
        pprint(log_dict)
        print("\n")
    accelerator.log(log_dict)
    accelerator.wait_for_everyone()

def print_model_state_size(model, model_path):
    '''
        Save Model's states and Print its size. 
    '''
    torch.save(model.state_dict(), model_path)
    fpath = model_path
    print("Size (MB):", os.path.getsize(fpath)/1e6)