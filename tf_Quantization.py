import torch
import argparse
from tqdm import tqdm

## .py imports ##
from tf_PointNet import * 
from read_point_cloud import * 
from utils import *

import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization  as tfmot

clean_nohup_out()
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--reduce_lr_wait', type=int, default=20)
parser.add_argument('--enc_dropout', type=float, default=0.2)
parser.add_argument('--dec_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--smaller_run', action="store_true")
parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--conv2lin', action="store_true")
parser.add_argument('--load_ver', help="Which version of model to load")
parser.add_argument('--debug_vis', action="store_true")
args = parser.parse_args()

# Initialize log file
ver = args.load_ver
pprint(vars(args))

# Model Load directory
model_pth = f"./{ver}/model.keras"
save_model_pth = f"./{ver}/quant_model.pth"
plot_name = f"./{ver}/quant_plot.png"

## Preprocess Data
if not os.path.exists("/data/train_X_y_ver_all_xyz_energy.pt"):
    raise FileNotFoundError("Make sure you download the data and put it inside a data folder")
elif not os.path.exists("data/preprocessed_data.pt"):
    from preprocess import preprocess
    preprocess("/data/train_X_y_ver_all_xyz_energy.pt")

# Load and preprocess data
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

# Initialize model within the strategy scope
@tf.function
def loss_fn(out, y, training=True):
    if training:
        xyz_loss = tf.keras.losses.MSE(y_true=y[:, :-1], y_pred=out[:, :-1])
        xyz_loss = tf.nn.compute_average_loss(xyz_loss)
        energy_loss = tf.keras.losses.MAE(y_true=y[:, -1], y_pred=out[:, -1])
        return xyz_loss + args.scale_energy_loss * energy_loss
    else:
        loss = tf.keras.losses.MSE(y_true=y, y_pred=out)
        return tf.reduce_mean(loss)
    
model = PointClassifier(
    n_hits=args.n_hits,
    dim=F_dim,
    out_dim=y.shape[-1],
    dim_reduce_factor=args.dim_reduce_factor,
    args=vars(args),
)

# def create_quantization_model(model):
#   layers = []
#   for i in range(len(model.layers)):
#     if isinstance(model.layers[i], tf.keras.models.Model):
#       quant_sub_model = tf.keras.models.clone_model(model.layers[i], clone_function=apply_quantization)
#       layers.append(tfmot.quantization.keras.quantize_apply(quant_sub_model))
#     else:
#       layers.append(apply_quantization(model.layers[i]))
#   quant_model = tf.keras.models.Sequential(layers)
#   return quant_model
# def apply_quantization(layer):
#   if isinstance(layer, tf.keras.layers.Dense):
#     return tfmot.quantization.keras.quantize_annotate_layer(layer)
#   return layer

## load model
model = tf.keras.models.load_model(model_pth, custom_objects={'PointClassifier': PointClassifier})
print(f"\nModel loaded from {model_pth}")

## Size before quant
# print("Time and Performance before quantizaing: \n")
# input_tensor = tf.keras.layers.Input(shape=(2126, 5))
# enc = model.layers[0](input_tensor)
# dec = model.layers[1](enc)
# combined_model = tf.keras.models.Model(inputs=input_tensor, outputs=dec)
# print(combined_model.summary())
# print("Size (MB):", os.path.getsize(model_pth)/1e6)

# enc = create_quantization_model(model.layers[0])
# print(enc.summary())

## Quantize
## TODO: Doesn't work for some reason.
quantize_model = tfmot.quantization.keras.quantize_model 
# print(dec)
# print()
# print(model.layers[1])
# qmodel = quantize_model(model.layers[1].layers)
# qmodel = quantize_model(model)
# print(qmodel.summary())
print([layer for layer in model.layers[0].layers])
# print([layer for layer in model.layers[1].layers])
print("Size (MB):", os.path.getsize(model_pth)/1e6)

# ## Size after quant
# print("\n\nTime and Performance after quantizaing: \n")
# # print(qmodel.summary())
# print(qmodel.layers[0].summary())
# print(qmodel.layers[1].summary())
# # pprint([layer for layer in qmodel.layers[0].layers])
# # pprint([layer for layer in qmodel.layers[1].layers])
# print(qmodel.save(save_model_pth))
# print("Size (MB):", os.path.getsize(save_model_pth)/1e6)

# ## Save model outputs and labels. Compute and store diff between output and label.
# epochs = range(args.epochs)
# diff = {"x":[], "y":[], "z":[], "radius": [], "unif_r":[], "energy":[]}
# dist = {"x":[], "y":[], "z":[], "x_pred":[], "y_pred":[], "z_pred":[], "energy":[], "energy_pred":[],
#          "radius": [], "radius_pred": [], "unif_r": [], "unif_r_pred": []}
# abs_diff = []
# with tqdm(total=len(test_loader), mininterval=5) as pbar:
#     test_loss = 0
#     print("Validating...")
#     for i, batch in enumerate(test_loader):
#         X, y = batch
#         out = model(X, training=False)
#         abs_diff.append(tf.abs(y - out))
#         test_loss += loss_fn(out, y, training=False)

#         diff_tensor = y - out
#         dist["x"].append(y[:, 0])
#         dist["y"].append(y[:, 1])
#         dist["z"].append(y[:, 2])

#         dist["x_pred"].append(out[:, 0])
#         dist["y_pred"].append(out[:, 1])
#         dist["z_pred"].append(out[:, 2])
        
#         diff["x"].append(diff_tensor[:, 0])
#         diff["y"].append(diff_tensor[:, 1])
#         diff["z"].append(diff_tensor[:, 2])

#         dist["energy"].append(y[:, 3])
#         dist["energy_pred"].append(out[:, 3])
#         diff["energy"].append(diff_tensor[:, 3])

#         pbar.update()
#     test_loss /= len(test_loader)
#     print(test_loss)

# abs_diff = tf.concat(abs_diff, axis=0)

# ## plot and save
# plot_reg(diff=diff, dist=dist, test_loss=test_loss, abs_diff=abs_diff, save_name=plot_name)