#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
# export LD_LIBRARY_PATH="/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/:/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/python3.10/site-packages/tensorrt/"
export LD_LIBRARY_PATH="/home/akann1w0w1ck/miniconda3/lib/:/home/akann1w0w1ck/miniconda3/lib/python3.10/site-packages/tensorrt/"

# Part g)
# CSDIS4 - Predicting High - Low
# CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1g_csdis4" --window_sample_count "3" --window_size "10" --window_shift "1" \
# --model_name "CSDIS4" --dataset_name "stocks_maxmin" --test_set_method "normal" \
# --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" --pred_metrics "mre_mae_rmse" \
# --pred_samples "5"  --mask_method "custom" --batch_size "8" \
# --batch_size_val "8" --eager --mixed_precision --conv_channels_first "0" --target_dim "380" \
# --start_prop_train "0.0" --end_prop_train "0.6" --start_prop_val "0.6" --end_prop_val "0.8" \
# --start_prop_test "0.8" --end_prop_test "1.0" --num_steps "50" --s4_lmax "22" --layers "4" \
#  --channels "64" --featureemb "128" --timeemb "16"  --diffusion_embedding_dim "128"  --learning_rate "5e-4" \
#  --weight_decay "1e-6"

CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1g_csdis4_stdscale_outpscale" --window_sample_count "3" --window_size "10" --window_shift "1" \
--model_name "CSDIS4" --dataset_name "stocks_maxmin" --test_set_method "normal" \
--diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" --pred_metrics "mre_mae_rmse" \
--pred_samples "5"  --mask_method "custom" --batch_size "8" \
--batch_size_val "8" --eager --mixed_precision --conv_channels_first "0" --target_dim "380" \
--start_prop_train "0.0" --end_prop_train "0.6" --start_prop_val "0.6" --end_prop_val "0.8" \
--start_prop_test "0.8" --end_prop_test "1.0" --num_steps "50" --s4_lmax "22" --layers "4" \
 --channels "64" --featureemb "128" --timeemb "16"  --diffusion_embedding_dim "128"  --learning_rate "5e-4" \
 --weight_decay "1e-6" --scaler "standard" --scale_output

# SSSDS4 - Predicting High - Low
# CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1g_sssds4" --window_sample_count "3" \
# --window_size "10" --window_shift "1" --model_name "SSSDS4" --dataset_name "stocks_maxmin" \
# --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" --test_set_method "normal" \
# --pred_metrics "mre_mae_rmse" --pred_samples "5"  --mask_method "custom"  --batch_size "8" \
# --batch_size_val "8" --num_res_layers "36" --in_channels "380" --out_channels "380" \
# --eager --mixed_precision --conv_channels_first "0" --T "200" --s4_lmax "12" \
# --diffusion_step_embed_dim_mid "512" --diffusion_step_embed_dim_out "512" \
# --diffusion_step_embed_dim_in "128" --res_channels "256" --skip_channels "256" --start_prop_train "0.0" \
# --end_prop_train "0.6" --start_prop_val "0.6" --end_prop_val "0.8" --start_prop_test "0.8" --end_prop_test "1.0" \
# --learning_rate "2e-4"

CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1g_sssds4_stdscale_outpscale" --window_sample_count "3" \
--window_size "10" --window_shift "1" --model_name "SSSDS4" --dataset_name "stocks_maxmin" \
--diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" --test_set_method "normal" \
--pred_metrics "mre_mae_rmse" --pred_samples "5"  --mask_method "custom"  --batch_size "8" \
--batch_size_val "8" --num_res_layers "36" --in_channels "380" --out_channels "380" \
--eager --mixed_precision --conv_channels_first "0" --T "200" --s4_lmax "12" \
--diffusion_step_embed_dim_mid "512" --diffusion_step_embed_dim_out "512" \
--diffusion_step_embed_dim_in "128" --res_channels "256" --skip_channels "256" --start_prop_train "0.0" \
--end_prop_train "0.6" --start_prop_val "0.6" --end_prop_val "0.8" --start_prop_test "0.8" --end_prop_test "1.0" \
--learning_rate "2e-4" --scaler "standard" --scale_output