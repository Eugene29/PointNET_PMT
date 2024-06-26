**Steps:**

0. Download data from `https://drive.google.com/file/d/1F_ZjeqmKkpWNXyd9JD-zty2SNe95TlUz/view?usp=sharing`
1. Add data called `train_X_y_ver_all_xyz_energy.pt` into `/data` folder
2. Install packages `pip install -r requirements.txt` (Should work but haven't tested. Let me know if it doesn't)
3. run `. multiple_exp.sh` (edit bash file configuration such as adding `--debug` to run **sample** of data)
4. See if you can replicate plots inside `example_plots` folder

**Notes:**
- Full training should take ~15 min (based on 4 x A5000)
- This code is agnostic of device (gpu, cpu, multi-gpu)
- Training log will print out in `nohup.out` & `{ver}/train.txt`

**Example Plots:**

Fully Trained Plot (XL)
`nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.1 --weight_decay 5e-3 --lr 1e-3 --save_ver XL \
                      --seed 999 --dim_reduce_factor 2 --batch_size 256 --scale_energy_loss 1000 --conv2lin`
![Example Image](example_plots/pointNET_hist.png)

Debug Mode Plot (XL debug)
`nohup accelerate launch train.py --use_wandb --epochs 600 --enc_dropout 0 --dec_dropout 0.1 --weight_decay 5e-3 --lr 1e-3 --save_ver XL \
                      --seed 999 --dim_reduce_factor 2 --batch_size 256 --scale_energy_loss 1000 --conv2lin --debug`
![Example Image](example_plots/debug_pointNET_hist.png)
