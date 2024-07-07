#!/bin/bash

## Train Model (PyTORCH)
# nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.1 --weight_decay 1e-4 --lr 1e-3 --save_ver XL \
#                       --seed 999 --dim_reduce_factor 2 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug
# nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0.01 --dec_dropout 0.01 --weight_decay 1e-5 --lr 1e-3 --save_ver L \
#                       --seed 999 --dim_reduce_factor 2.5 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug
# nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.01 --weight_decay 1e-4 --lr 1e-3 --save_ver M \
#                       --seed 999 --dim_reduce_factor 3 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug
# nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.01 --weight_decay 5e-6 --lr 1e-3 --save_ver S \
#                       --seed 999 --dim_reduce_factor 3.5 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug

# ## Visualize
# nohup accelerate launch vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver XL1 --conv2lin --dim_reduce_factor 2
# nohup accelerate launch vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver L --conv2lin --dim_reduce_factor 2.5
# nohup accelerate launch vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver M --conv2lin --dim_reduce_factor 3
# nohup accelerate launch vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver S --conv2lin --dim_reduce_factor 3.5

# # Quantization
# nohup python Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 2 --load_ver XL --conv2lin
# nohup python Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 2.5 --load_ver L --conv2lin
# nohup python Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 3 --load_ver M --conv2lin
# nohup python Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 3.5 --load_ver S --conv2lin

## Train Model (Tensorflow)
# nohup python tf_train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.1 --weight_decay 1e-4 --lr 1e-3 --save_ver tf_XL \
#                       --seed 999 --dim_reduce_factor 2 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug
# nohup python tf_train.py --use_wandb --epochs 600 --enc_dropout 0.01 --dec_dropout 0.01 --weight_decay 1e-5 --lr 1e-3 --save_ver tf_L \
#                       --seed 999 --dim_reduce_factor 2.5 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug
# nohup python tf_train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0 --weight_decay 0 --lr 1e-3 --save_ver tf_M \
#                       --seed 999 --dim_reduce_factor 3 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug
# nohup python tf_train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0 --weight_decay 0 --lr 1e-3 --save_ver tf_S \
#                       --seed 999 --dim_reduce_factor 3.5 --batch_size 256 --scale_energy_loss 1000 --conv2lin #--debug

# ## Visualize
# nohup python -u tf_vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver tf_XL --conv2lin --dim_reduce_factor 2
# nohup python -u tf_vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver tf_L --conv2lin --dim_reduce_factor 2.5
# nohup python -u tf_vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver tf_M --conv2lin --dim_reduce_factor 3
# nohup python -u tf_vis_for_pointNET.py --batch_size 256 --seed 999  --load_ver tf_S --conv2lin --dim_reduce_factor 3.5

# # Quantization
# nohup python tf_Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 2 --load_ver XL --conv2lin
# nohup python tf_Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 2.5 --load_ver L --conv2lin
# nohup python tf_Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 3 --load_ver M --conv2lin
# nohup python tf_Quantization.py  --seed 999 --batch_size 256 --dim_reduce_factor 3.5 --load_ver S --conv2lin
