#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH="/home/akann1w0w1ck/miniconda3/lib/:/home/akann1w0w1ck/miniconda3/lib/python3.10/site-packages/tensorrt/"

# SSSD-SA  20%RM
python3 -O train.py --exp_name "ptbxl248_ssdsa_rm_run1" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 --test_only

# SSSD-SA  20%MNR
python3 -O train.py --exp_name "ptbxl248_ssdsa_mnr_run1" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 --test_only

# SSSD-SA  20%BM
python3 -O train.py --exp_name "ptbxl248_ssdsa_bm_run1" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 --test_only

# SSSD-SA  20%RM Run2
python3 -O train.py --exp_name "ptbxl248_ssdsa_rm_run2" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100


# SSSD-SA  20%MNR Run2
python3 -O train.py --exp_name "ptbxl248_ssdsa_mnr_run2" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 

# SSSD-SA  20%BM Run2
python3 -O train.py --exp_name "ptbxl248_ssdsa_bm_run2" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 

# SSSD-SA  20%RM Run3
python3 -O train.py --exp_name "ptbxl248_ssdsa_rm_run3" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100  

# SSSD-SA  20%MNR Run3
python3 -O train.py --exp_name "ptbxl248_ssdsa_mnr_run3" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 




# SSSD-SA  20%BM Run3
python3 -O train.py --exp_name "ptbxl248_ssdsa_bm_run3" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "96" --batch_size_val "640" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first 1 --T 100 

