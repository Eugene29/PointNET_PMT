import torch
import argparse
from tqdm import tqdm
from contextlib import redirect_stdout
import io

## .py imports ##
from tf_PointNet import * 
from read_point_cloud import * 
from utils import *

import tensorflow as tf
import tf_keras
# from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

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
init_logfile(ver, mode="quantize")
pprint(vars(args))

# Model Load directory
model_pth = f"./{ver}/model.keras"
save_model_pth = f"./{ver}/quant_model.keras"
plot_name = f"./{ver}/quant_plot.png"

## Preprocess Data
if not os.path.exists("data/train_X_y_ver_all_xyz_energy.pt"):
    raise FileNotFoundError("Make sure you download the data and put it inside a data folder")
elif not os.path.exists("data/preprocessed_data.pt"):
    from preprocess import preprocess
    preprocess("data/train_X_y_ver_all_xyz_energy.pt")

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
        xyz_loss = tf_keras.losses.MSE(y_true=y[:, :-1], y_pred=out[:, :-1])
        xyz_loss = tf.nn.compute_average_loss(xyz_loss)
        energy_loss = tf_keras.losses.MAE(y_true=y[:, -1], y_pred=out[:, -1])
        return xyz_loss + args.scale_energy_loss * energy_loss
    else:
        loss = tf_keras.losses.MSE(y_true=y, y_pred=out)
        return tf.reduce_mean(loss)
    
## load model
from tf_keras.utils import custom_object_scope

model = tf_keras.models.load_model(model_pth, custom_objects={"TorchDefaultLinInit": TorchDefaultLinInit})
print(f"\nModel loaded from {model_pth}")


## Quantize
## Warm up Kernel
with redirect_stdout(io.StringIO()): ## Suppressing prints
    tf_pred_all_data(model, train_loader, val_loader, test_loader, loss_fn)

## Before Quantization
print(model.summary())
print("Size (MB):", os.path.getsize(model_pth)/1e6) ## size
tf_pred_all_data(model, train_loader, val_loader, test_loader, loss_fn)

## After Quantization (Qkeras VERSION)
quantize_model = tfmot.quantization.keras.quantize_model 
with custom_object_scope({'TorchDefaultLinInit': TorchDefaultLinInit}):
    qmodel = quantize_model(model)
    qmodel.compile()
print(qmodel.summary())
# with redirect_stdout(io.StringIO()): ## Suppressing prints
#     tf_pred_all_data(qmodel, train_loader, val_loader, test_loader, loss_fn)
tf_pred_all_data(qmodel, train_loader, val_loader, test_loader, loss_fn)
qmodel.save(save_model_pth)
print("Size (MB):", os.path.getsize(save_model_pth)/1e6) ## size

## After Quantization (tflite VERSION)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Ensure that if your model uses any specific ops that are not typical, they are supported in TFLite
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # Ensure model inputs are quantized to int8
# converter.inference_output_type = tf.int8  # Ensure model outputs are quantized to int8
# converter.representative_dataset = representative_data_gen
# converter.experimental_new_converter = True

# # Convert the model
# print("converting into tf_lite...")
# tflite_model = converter.convert()
# # print(tflite_model.summary())

# # Save the TensorFlow Lite model to disk
# tf_model_pth = f'{ver}/qmodel.tflite'
# with open(tf_model_pth, 'wb') as f:
#     f.write(tflite_model)
# print("Size (MB):", os.path.getsize(tf_model_pth)/1e6) ## size
# # tf_pred_all_data(tflite_model, train_loader, val_loader, test_loader, loss_fn)

# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path=tf_model_pth)
# input_data = next(iter(test_loader))[0].numpy() # or np.int8 if your model uses int8 inputs
# # input_data = np.expand_dims(input_data, axis=0) # Shape the data as (1, 224, 224, 3)
# interpreter.resize_tensor_input(0, input_data.shape)
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# class tflite_inference():
#     def __init__(self, interpreter, input_details, output_details):
#         self.interpreter = interpreter
#         self.input_details = input_details
#         self.output_details = output_details

#     def __call__(self, inputs, training):
#         # Set the tensor to point to the input data to be inferred
#         self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
#         self.interpreter.invoke()

#         # Extract the output data
#         output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
#         return output_data

# tflite_inf = tflite_inference(interpreter, input_details, output_details)
# tf_pred_all_data(tflite_inf, train_loader, val_loader, test_loader, loss_fn, verbose=True)


## PLOT
## Save model outputs and labels. Compute and store diff between output and label.
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