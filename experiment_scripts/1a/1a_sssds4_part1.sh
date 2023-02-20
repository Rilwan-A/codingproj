#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH="/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/:/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/python3.10/site-packages/tensorrt/"

# SSSD-S4  20%RM
python3 -O train.py --exp_name "ptbxl248_sssds4_rm_run1" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "14" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "2e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "2" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256"

# SSSD-S4  20%MNR
python3 -O train.py --exp_name "ptbxl248_sssds4_mnr_run1" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
    --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "14" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "2e-4" \
    --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "2" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 

# # SSSD-S4  20%BM
# python3 -O train.py --exp_name "ptbxl248_sssds4_bm_run1" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 


# # SSSD-S4  20%RM Run2
# python3 -O train.py --exp_name "ptbxl248_sssds4_rm_run2" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 

# # SSSD-S4  20%MNR Run0
# python3 -O train.py --exp_name "ptbxl248_sssds4_mnr_run2" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 

# # SSSD-S4  20%BM Run2
# python3 -O train.py --exp_name "ptbxl248_sssds4_bm_run2" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 


# # SSSD-S4  20%RM Run3
# python3 -O train.py --exp_name "ptbxl248_sssds4_rm_run3" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "rm" --missing_k "52" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256"  

# # SSSD-S4  20%MNR Run3
# python3 -O train.py --exp_name "ptbxl248_sssds4_mnr_run3" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "mnr" --missing_k "5" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 

# # SSSD-S4  20%BM Run3
# python3 -O train.py --exp_name "ptbxl248_sssds4_bm_run3" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
#     --pred_metrics "mae_rmse_crps" --pred_samples "5"  --mask_method "bm" --missing_k "5" --batch_size "6" --batch_size_val "100" --dataset_shape "D_c_seq" --in_channels "12" -out_channels "12" --learning_rate "6e-4" \
#     --steps_per_epoch 0.5 --eager --mixed_precision --conv_channels_first "0" --T "100" --grad_accum "5" --s4_lmax "250" --diffusion_step_embed_dim_mid "256" --diffusion_step_embed_dim_out "256" 

