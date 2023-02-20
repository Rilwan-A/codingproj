#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH="/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/:/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/python3.10/site-packages/tensorrt/"
#export LD_LIBRARY_PATH="/home/akann1w0w1ck/miniconda3/lib/:/home/akann1w0w1ck/miniconda3/lib/python3.10/site-packages/tensorrt/"

# SSSD-S4  20%RM
python3 -O train.py --exp_name "ptbxl248_csdis4_rm_run1" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "2e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "2" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"

# SSSD-S4  20%MNR
python3 -O train.py --exp_name "ptbxl248_csdis4_mnr_run1" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "2e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "2" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  

# SSSD-S4  20%BM
python3 -O train.py --exp_name "ptbxl248_csdis4_bm_run1" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  


# SSSD-S4  20%RM Run2
python3 -O train.py --exp_name "ptbxl248_csdis4_rm_run2" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  

# SSSD-S4  20%MNR Run2
python3 -O train.py --exp_name "ptbxl248_csdis4_mnr_run2" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  

# SSSD-S4  20%BM Run2
python3 -O train.py --exp_name "ptbxl248_csdis4_bm_run2" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  


# SSSD-S4  20%RM Run3
python3 -O train.py --exp_name "ptbxl248_csdis4_rm_run3" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"   

# SSSD-S4  20%MNR Run3
python3 -O train.py --exp_name "ptbxl248_csdis4_mnr_run3" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  

# SSSD-S4  20%BM Run3
python3 -O train.py --exp_name "ptbxl248_csdis4_bm_run3" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "32"  --batch_size_val "320"  --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --num_steps "50" --grad_accum "5" --s4_lmax "250" --layers "4" --channels "64" --featureemb "128" --timeemb "16"  "diffusion_embedding_dim" "128"  

