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
import tensorflow as tf

def init_logfile(i, mode=None):
    '''
        create and set logfile to be written. Also write init messages such as args and seed
    '''
    save_dir = f"./{i}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fname = "train.txt" if mode is None else f"{mode}.txt" 
    log_file = open(f"{i}/{fname}", 'w', buffering=1)
    sys.stderr = log_file
    sys.stdout = log_file
    return save_dir, log_file

def clean_nohup_out():
    with open("nohup.out", "w") as f:
        pass

def preprocess_features(X, n_hits):
    '''
        1. swap the syntax (time, charge, x, y, z) -> (x, y, z, time, charge)
        2. zero out xyz positions of sensors that are not activated
        TODO: can make it more efficient.
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
    # new_X = new_X.mT if args.conv2lin else new_X
    new_X = new_X.mT
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
    # set_seed(seed=args.seed)
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

def tf_shuffle_data(X, y, n_data, args):
    '''
        shuffle data and return train, val, test datasets in TensorFlow
    '''
    # Set seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Shuffle indices
    idx = np.random.permutation(n_data)
    X, y = tf.gather(X, idx), tf.gather(y, idx)

    # Set up prefetching
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Create datasets
    train_split = 0.6
    val_split = 0.2
    train_idx = int(n_data * train_split)
    val_idx = int(train_idx + n_data * val_split)

    # torch.save((torch.tensor(X[:train_idx].numpy()), torch.tensor(y[:train_idx].numpy())), "data/preprocessed_train.pt")
    # torch.save((torch.tensor(X[train_idx:val_idx].numpy()), torch.tensor(y[train_idx:val_idx].numpy())), "data/preprocessed_val.pt")
    # torch.save((torch.tensor(X[val_idx:].numpy()), torch.tensor(y[val_idx:].numpy())), "data/preprocessed_test.pt")

    train_dataset = tf.data.Dataset.from_tensor_slices((X[:train_idx], y[:train_idx]))
    val_dataset = tf.data.Dataset.from_tensor_slices((X[train_idx:val_idx], y[train_idx:val_idx]))
    test_dataset = tf.data.Dataset.from_tensor_slices((X[val_idx:], y[val_idx:]))
    print(f"len(train_dataset): {len(train_dataset)}")

    # Batch datasets
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(AUTOTUNE)
    val_dataset = val_dataset.batch(args.batch_size).prefetch(AUTOTUNE)
    test_dataset = test_dataset.batch(args.batch_size).prefetch(AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

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
    
def tf_plot_reg(diff, dist, test_loss, abs_diff, save_name):
    '''
        Create and save a plot of 6 total histogram (1 fig). 
    '''
    abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = abs_diff.numpy().mean(axis=0)
    energy_diff = tf.concat(diff["energy"], axis=0).numpy()
    energy_pred = tf.concat(dist["energy_pred"], axis=0).numpy()
    energy = tf.concat(dist["energy"], axis=0).numpy()

    x_diff = tf.concat(diff["x"], axis=0).numpy()
    y_diff = tf.concat(diff["y"], axis=0).numpy()
    z_diff = tf.concat(diff["z"], axis=0).numpy()

    x_pred = tf.concat(dist["x_pred"], axis=0).numpy()
    y_pred = tf.concat(dist["y_pred"], axis=0).numpy()
    z_pred = tf.concat(dist["z_pred"], axis=0).numpy()

    x = tf.concat(dist["x"], axis=0).numpy()
    y = tf.concat(dist["y"], axis=0).numpy()
    z = tf.concat(dist["z"], axis=0).numpy()

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

def tf_pred_all_data(model, train_loader, val_loader, test_loader, loss_fn, verbose=False):
    '''
        Predict on all data (train, val, test), print each loss, and the time it took.
    '''
    strt = time()
    train_loss, val_loss, test_loss = 0, 0, 0
    abs_diff = []
    print("testing inference speed and performance...")
    for X, y in train_loader:
        out = model(inputs=X, training=False)
        loss = loss_fn(out=out, y=y, training=False)
        train_loss += loss.numpy()
        abs_diff.append(tf.abs(y - out))
        if verbose:
            print(out)

    for X, y in val_loader:
        out = model(inputs=X, training=False)
        loss = loss_fn(out=out, y=y, training=False)
        val_loss += loss.numpy()
        abs_diff.append(tf.abs(y - out))

    for X, y in test_loader:
        out = model(inputs=X, training=False)
        loss = loss_fn(out=out, y=y, training=False)
        test_loss += loss.numpy()
        abs_diff.append(tf.abs(y - out))
        # test_loss += distributed_eval_step(X, y).numpy()

    train_loss /= len(val_loader) # test_len
    val_loss /= len(test_loader) # test_len
    test_loss /= len(test_loader) # test_len

    ## Logging
    abs_diff = tf.concat(abs_diff, axis=0)
    abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = abs_diff.cpu().numpy().mean(axis=0)
    abs_dict = {"abs_x_diff": abs_x_diff, "abs_y_diff": abs_y_diff, "abs_z_diff": abs_z_diff, "abs_energy_diff": abs_energy_diff}
    print(f"test_loss: {test_loss:.2f}")

    # Logging results
    tot_time = time() - strt
    print(f"Entire script time taken: {tot_time:.0f} secs")

    # Summary of results
    log_dict = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }

    log_dict.update(abs_dict)

    print("\nResults summary:")
    for k, v in log_dict.items():
        print(f"{k}: {round(v, 2)}")


def print_model_state_size(model, model_pth):
    '''
        Save Model's states and Print its size. 
    '''
    torch.save(model.state_dict(), model_pth)
    print("Size (MB):", os.path.getsize(model_pth)/1e6)

def build_model(model):
    dense_input_shapes = model.layers[0].encoder_input_shapes
    # all_input_shapes = dense_input_shapes + [dense_input_shapes[1], dense_input_shapes[2]]
    # for layer, input_shape in zip(model.layers[0].layers, all_input_shapes):
    #     layer.build((input_shape,))

    input_dim = dense_input_shapes[0]
    latent_dim = model.layers[0].latent_dim
    n_hits = 2126
    # model.layers[0].build((None, n_hits, input_dim))
    # model.layers[1].build((None, latent_dim, ))
    model.build((None, n_hits, input_dim))

    print("\n\nEntire Model Summary: ")
    print(model.summary())
    # print("\n\nEncoder Model Summary: ")
    print(model.layers[0].layers)
    # print(model.layers[0].summary())
    # print("\n\nDecoder Model Summary: ")
    print(model.layers[1].layers)
    # print(model.layers[1].summary())
    # pprint(model.layers[0].layers + model.layers[1].layers)

def flatten_model(model_nested):
    import tf_keras
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    # model_flat = keras.models.Sequential(layers_flat)
    model_flat = tf_keras.models.Sequential(layers_flat)
    return model_flat

# def flatten_model(model_nested):
#     def get_layers(layers):
#         layers_flat = []
#         for layer in layers:
#             try:
#                 layers_flat.extend(get_layers(layer.layers))
#             except AttributeError:
#                 layers_flat.append(layer)
#         return layers_flat

#     # model_flat = tf.keras.models.Sequential(
#     model_flat = tf_keras.models.Sequential(
#         get_layers(model_nested.layers)
#     )
#     return model_flat