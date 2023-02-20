#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# ptbxl 248 
# - CSDI-S4, SSSD-SA, SSSD-S4
# - Missingess factors: 20%RM, 20%MNR, 20%BM
# - D1 (diffusion only to part to be imputed) , D0 (diffusion to entire sample) [ignored]
# - MAE, RMSE, MRE avg over 10 generated samples per test datum over 3 trials

# CSDI-S4  20%RM
# python3 train.py --exp_name "ptbxl248_csdis4_rm_run1" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"
# python3 train.py --exp_name "ptbxl248_csdis4_rm_run2" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"
# python3 train.py --exp_name "ptbxl248_csdis4_rm_run3" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"

# # CSDI-S4  20%MNR
# python3 train.py --exp_name "ptbxl248_csdis4_mnr_run1" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"
# python3 train.py --exp_name "ptbxl248_csdis4_mnr_run2" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"
# python3 train.py --exp_name "ptbxl248_csdis4_mnr_run3" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"

# # CSDI-S4  20%BM
# python3 train.py --exp_name "ptbxl248_csdis4_bm_run1" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"
# python3 train.py --exp_name "ptbxl248_csdis4_bm_run2" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"
# python3 train.py --exp_name "ptbxl248_csdis4_bm_run3" --model_name "CSDIS4" --dataset_name "ptbxl_248" --diffusion_method "csdi" --epochs "200" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "1e-3"\
#     --weight_decay "1e-6"


# SSSD-SA  20%RM
python3 train.py --exp_name "ptbxl248_ssdsa_rm_run1" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 100 --validation_steps 20
python3 train.py --exp_name "ptbxl248_ssdsa_rm_run2" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200
python3 train.py --exp_name "ptbxl248_ssdsa_rm_run3" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200


# SSSD-SA  20%MNR
python3 train.py --exp_name "ptbxl248_ssdsa_mnr_run1" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200

python3 train.py --exp_name "ptbxl248_ssdsa_mnr_run2" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200

python3 train.py --exp_name "ptbxl248_ssdsa_mnr_run3" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200


# SSSD-SA  20%BM
python3 train.py --exp_name "ptbxl248_ssdsa_bm_run1" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200

python3 train.py --exp_name "ptbxl248_ssdsa_bm_run2" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200

python3 train.py --exp_name "ptbxl248_ssdsa_bm_run3" --model_name "SSSDSA" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
    --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4" \
    --steps_per_epoch 1000 --validation_steps 200




# # SSSD-S4  20%RM
# python3 train.py --exp_name "ptbxl248_ssds4_rm_run1" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"
# python3 train.py --exp_name "ptbxl248_ssds4_rm_run2" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"
# python3 train.py --exp_name "ptbxl248_ssds4_rm_run3" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "rm" --missing_k "52" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"

# # SSSD-S4  20%MNR
# python3 train.py --exp_name "ptbxl248_ssds4_mnr_run1" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"
# python3 train.py --exp_name "ptbxl248_ssds4_mnr_run2" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"
# python3 train.py --exp_name "ptbxl248_ssds4_mnr_run3" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "mnr" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"

# # SSSD-S4  20%BM
# python3 train.py --exp_name "ptbxl248_ssds4_bm_run1" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"
# python3 train.py --exp_name "ptbxl248_ssds4_bm_run2" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"
# python3 train.py --exp_name "ptbxl248_ssds4_bm_run3" --model_name "SSSDS4" --dataset_name "ptbxl_248" --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse_mre" \
#     --pred_metrics "mae_rmse_mre" --pred_samples "10"  --mask_method "bm" --missing_k "5" --batch_size "32" --batch_size_val "96" --dataset_shape "D_c_seq" --target_dim "12" --learning_rate "2e-4"

# TODO: Forecasting
# - CSDI-S4, Solar
# - Missingess factors: 
# - MSE, CRPS

