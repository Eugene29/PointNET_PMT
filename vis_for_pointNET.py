import torch
import accelerate
from accelerate import Accelerator
import argparse
from tqdm import tqdm
from optimum.quanto import qint8
from optimum.quanto.quantize import quantize


## .py imports ##
from PointNet import * 
from read_point_cloud import * 
from utils import *

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

accelerator = accelerate.Accelerator()
ddp_kwargs = accelerate.DistributedDataParallelKwargs(broadcast_buffers=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

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
model, train_loader, val_loader, test_loader = accelerator.prepare(model, train_loader, val_loader, test_loader)

## Load Model
if args.load_ver == "Quantized":
    ## UNAVAILABLE ATM
    quantize(model, weights=qint8, activations=qint8)
    save_name = f"Quantization/Quantized/pointNET"
    model_dir = f"Quantization/Quantized/model.pth"
    state_dict = torch.load(model_dir, map_location=accelerator.device)
    model.module.load_state_dict(state_dict)
else:
    save_name = f"{args.load_ver}/pointNET"
    model_dir = f"{args.load_ver}/model_dir/"
    accelerator.load_state(model_dir)


## Save model outputs and labels. Compute and store diff between output and label.
epochs = range(args.epochs)
diff = {"x":[], "y":[], "z":[], "radius": [], "unif_r":[], "energy":[]}
dist = {"x":[], "y":[], "z":[], "x_pred":[], "y_pred":[], "z_pred":[], "energy":[], "energy_pred":[],
         "radius": [], "radius_pred": [], "unif_r": [], "unif_r_pred": []}
abs_diff = []
model.eval()
with tqdm(total=len(test_loader), mininterval=5) as pbar, torch.no_grad():
    test_loss = 0

    accelerator.print("Validating...")
    for i, batch in enumerate(test_loader):
        X, y = batch
        out = model(X)
        out, y = accelerator.gather(out), accelerator.gather(y)
        abs_diff.append((y - out).abs())
        test_loss += F.mse_loss(out, y).item()

        diff_tensor = y - out
        dist["x"].append(y[:, 0])
        dist["y"].append(y[:, 1])
        dist["z"].append(y[:, 2])

        dist["x_pred"].append(out[:, 0])
        dist["y_pred"].append(out[:, 1])
        dist["z_pred"].append(out[:, 2])
        
        diff["x"].append(diff_tensor[:, 0])
        diff["y"].append(diff_tensor[:, 1])
        diff["z"].append(diff_tensor[:, 2])

        dist["energy"].append(y[:, 3])
        dist["energy_pred"].append(out[:, 3])
        diff["energy"].append(diff_tensor[:, 3])

        pbar.update()
    test_loss /= len(test_loader)
    accelerator.print(test_loss)

dist, diff, abs_diff = accelerator.gather(dist), accelerator.gather(diff), accelerator.gather(abs_diff)
abs_diff = torch.cat(abs_diff)

## plot and save
plot_reg(diff=diff, dist=dist, test_loss=test_loss, abs_diff=abs_diff, save_name=save_name)