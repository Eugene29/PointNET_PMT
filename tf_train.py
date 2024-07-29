import tensorflow as tf
import tf_keras
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from time import time

from tf_PointNet import *
from qPointNet import qPointClassifier
from utils import *
from preprocess import preprocess

# Start script timing
strt = time()
clean_nohup_out()

# Hyperparameters
parser = argparse.ArgumentParser()
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
parser.add_argument('--conv2lin', action="store_true")
parser.add_argument('--scale_energy_loss', type=float, default=1)
parser.add_argument('--QAT', action="store_true")

args = parser.parse_args()
ver = args.save_ver

# Set up TensorFlow distributed strategy
# strategy = tf.distribute.MirroredStrategy()
# args.num_devices = strategy.num_replicas_in_sync
# print(f"Number of devices: {args.num_devices}")

# Initialize W&B
if args.use_wandb and not args.debug:
    wandb.init(project="pointNET", config=args)
    config = wandb.config

# Initialize log file
init_logfile(ver) if not args.QAT else init_logfile(ver, mode="QAT")
pprint(vars(args))

# Model save directory
model_pth = f"./{ver}/model.keras"

# Save preprocess data
if not os.path.exists("data/train_X_y_ver_all_xyz_energy.pt"):
    raise FileNotFoundError("Make sure you download the data and put it inside a data folder")
elif not os.path.exists("data/preprocessed_data.pt"):
    preprocess("data/train_X_y_ver_all_xyz_energy.pt")

# Load Preprocess data
print("loading data...")
X, y = torch.load("data/preprocessed_data.pt")
X = tf.convert_to_tensor(X.numpy(), dtype=tf.float32)
y = tf.convert_to_tensor(y.numpy(), dtype=tf.float32)
if args.debug:
    print("debug got called")
    small = 5000
    X, y = X[:small], y[:small]


# Update batch size
n_data, args.n_hits, F_dim = X.shape

# Shuffle data with seed
train_loader, val_loader, test_loader = tf_shuffle_data(X=X, y=y, n_data=n_data, args=args)
train_len, val_len, test_len = len(train_loader), len(val_loader), len(test_loader)

## initialize distributed dataset
# train_loader = strategy.experimental_distribute_dataset(train_loader)
# val_loader = strategy.experimental_distribute_dataset(val_loader)
# test_loader = strategy.experimental_distribute_dataset(test_loader)

# Initialize TensorBoard
# import datetime
# log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# summary_writer = tf.summary.create_file_writer(log_dir)
    
# Initialize model within the strategy scope
# with strategy.scope():
if args.QAT:
    model = qPointClassifier(
        n_hits=args.n_hits,
        dim=F_dim,
        out_dim=y.shape[-1],
        dim_reduce_factor=args.dim_reduce_factor,
        args=vars(args),
    )
else:
    nested_model = PointClassifier(
        n_hits=args.n_hits,
        dim=F_dim,
        out_dim=y.shape[-1],
        dim_reduce_factor=args.dim_reduce_factor,
        args=vars(args),
    )
    # model = flatten_model(nested_model)
    model = nested_model
    build_model(model)
    print(model.summary())

# Define optimizer and loss function
optimizer = tf_keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay, epsilon=1e-8)

# Define learning rate scheduler
lr_scheduler = tf_keras.callbacks.ReduceLROnPlateau(
    monitor='train_loss',
    factor=0.8,
    # patience=int(args.patience * strategy.num_replicas_in_sync), ## for distributed training 
    min_lr=1e-5,
    min_delta=1.5e-1,
    mode='min',
    verbose=1,
)
lr_scheduler.set_model(model)

# Compile model
model.compile(optimizer=optimizer)

## Count param after running one sample
nparam = model.count_params()
print(f"num. parameters: {nparam}\n")

# Training loop
best_val, best_train = float("inf"), float("inf")
train_lst, val_time_lst = [], []

## Loss Function
@tf.function
def loss_fn(out, y, training=True):
    if training:
        xyz_loss = tf_keras.losses.MSE(y_true=y[:, :-1], y_pred=out[:, :-1])
        xyz_loss = tf.nn.compute_average_loss(xyz_loss)
        energy_loss = tf_keras.losses.MAE(y_true=y[:, -1], y_pred=out[:, -1])
        return xyz_loss + args.scale_energy_loss * energy_loss
    else:
        # tf.print("evaluating")
        loss = tf_keras.losses.MSE(y_true=y, y_pred=out)
        return tf.reduce_mean(loss)

## For distributed training
# @tf.function
# def train_step(X, y):
#     with tf.GradientTape() as tape:
#         out = model(x=X, training=True)
#         loss = loss_fn(out=out, y=y, training=True)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

# @tf.function
# def distributed_train_step(X, y):
#     per_replica_losses = strategy.run(train_step, args=(X, y))
#     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# @tf.function
# def eval_step(X, y):
#     out = model(x=X, training=False)
#     loss = loss_fn(out=out, y=y, training=False)
#     return loss

# @tf.function
# def distributed_eval_step(X, y):
#     per_replica_losses = strategy.run(eval_step, args=(X, y))
#     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Start profiling
# tf.profiler.experimental.start(log_dir)

pbar = tqdm(total=args.epochs, mininterval=10)
mod = 5
for epoch in range(args.epochs):
    train_loss = 0

    for i, (X, y) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            out = model(x=X, training=True)
            loss = loss_fn(out=out, y=y, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss += loss.numpy()

    train_loss /= train_len
    train_lst.append(train_loss)
    current_lr = optimizer.learning_rate.numpy()

    # Manually call the ReduceLROnPlateau scheduler
    lr_scheduler.on_epoch_end(epoch, logs={'train_loss': train_loss})

    ## Validation
    if epoch % mod == 0 or epoch == args.epochs - 1:
        val_loss = 0
        val_strt = time()
        for X, y in val_loader:
            out = model(x=X, training=False)
            loss = loss_fn(out=out, y=y, training=False)
            val_loss += loss.numpy()
            # val_loss += distributed_eval_step(X, y).numpy() ## For distributed training
        val_loss /= val_len

        # Validation
        val_time_lst.append(time() - val_strt)

        # Save best model
        if val_loss < best_val and epoch > args.epochs / 2:
            best_val = val_loss
            print(f"New Best Score!! Best val loss: {best_val}")
            model.save(model_pth)

        TO_SKIP = 2
        if len(val_time_lst) > TO_SKIP:
            print(f"average val epoch time: {np.mean(val_time_lst[TO_SKIP:]):.4f}")
        pbar.update(mod)
        ## Prints
        print(f"\nEpoch {epoch}: ")
        print(f"Train Loss: {train_loss:.2f}\tval loss: {val_loss:.2f}\tCurrent LR: {current_lr:.6f}")
        print(f"Min Train Loss: {round(min(train_lst), 2)}")
        wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})
    else:
        print(f"Min Train Loss: {round(min(train_lst), 2)}")
        wandb.log({"Train Loss": train_loss})

# Stop profiling
# tf.profiler.experimental.stop()

# Load best model for testing
# model = tf_keras.models.load_model(model_pth, custom_objects={'PointClassifier': PointClassifier})
model = tf_keras.models.load_model(model_pth, custom_objects={"TorchDefaultLinInit": TorchDefaultLinInit})
print(f"\nModel loaded from {model_pth}")

# Test loop
test_loss = 0
abs_diff = []
for X, y in test_loader:
    out = model(x=X, training=False)
    loss = loss_fn(out=out, y=y, training=False)
    test_loss += loss.numpy()
    abs_diff.append(tf.abs(y - out))
    # test_loss += distributed_eval_step(X, y).numpy()

## Logging
abs_diff = tf.concat(abs_diff, axis=0)
abs_x_diff, abs_y_diff, abs_z_diff, abs_energy_diff = abs_diff.cpu().numpy().mean(axis=0)
abs_dict = {"abs_x_diff": abs_x_diff, "abs_y_diff": abs_y_diff, "abs_z_diff": abs_z_diff, "abs_energy_diff": abs_energy_diff}

test_loss /= test_len
print(f"test_loss: {test_loss:.2f}")

# Logging results
tot_time = time() - strt
print(f"Entire script time taken: {tot_time:.0f} secs")

# Summary of results
min_train = round(min(train_lst), 2)
log_dict = {
    "min_train": min_train,
    "min_val": best_val,
    "test_loss": test_loss,
    "avg_val_time": np.mean(val_time_lst) if len(val_time_lst) > 0 else 0,
}

log_dict.update(abs_dict)

if args.use_wandb and not args.debug:
    wandb.log(log_dict)
    wandb.finish()

print("\nResults summary:")
for k, v in log_dict.items():
    print(f"{k}: {round(v, 2)}")
