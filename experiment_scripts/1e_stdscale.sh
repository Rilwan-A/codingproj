#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
# export LD_LIBRARY_PATH="/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/:/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/python3.10/site-packages/tensorrt/"
export LD_LIBRARY_PATH="/home/akann1w0w1ck/miniconda3/lib/:/home/akann1w0w1ck/miniconda3/lib/python3.10/site-packages/tensorrt/"

               

# Part e)

  # CSDIS4 - Return prediction pn holidays across whole dataset with standard scaling and ouput scaling
CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1e_csdis4_stdscale_outpscale" \
 --window_sample_count "3" --window_size "20" --window_shift "5" --model_name "CSDIS4" \
 --dataset_name "stocks" --test_set_method "holidays_only" \
 --diffusion_method "csdi" --n_iters "150000" --val_metrics "mae_rmse" --pred_metrics "logreturn" \
  --pred_samples "5"  --mask_method "bm_channelgroup" --missing_k "0 28 29 102 103 123" --batch_size "8"\
 --batch_size_val "8" --eager --mixed_precision --conv_channels_first "0" --target_dim "126" \
 --num_steps "50" --s4_lmax "22" --layers "4" --channels "64" --featureemb "128" --timeemb "16" \
  --diffusion_embedding_dim "128" --learning_rate "1e-4" --weight_decay "1e-6" --patience "30" \
  --scaler "standard" --scale_output

# SSSDS4 - Return prediction with standard scaling and output scaling
CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1e_sssds4_stdscale_outpscale" \
--window_sample_count "3" --window_size "20" --window_shift "5" --model_name "SSSDS4" --dataset_name "stocks" \
 --diffusion_method "alvarez" --n_iters "150000" --val_metrics "mae_rmse" \
 --test_set_method "holidays_only" --pred_metrics "logreturn" --pred_samples "5" \
 --mask_method "bm_channelgroup" --missing_k "0 28 29 102 103 123" --batch_size "8" \
 --batch_size_val "8" --num_res_layers "32"  --in_channels "126" --out_channels "126" \
  --mixed_precision --eager --conv_channels_first "0" --T "200" --s4_lmax "22" \
   --diffusion_step_embed_dim_mid "512" --diffusion_step_embed_dim_out "512" \
    --diffusion_step_embed_dim_in "128" --res_channels "256" \
   --skip_channels "256" --learning_rate "1e-5" --patience "30" weight_decay "1e-6" \
   --scaler "standard" --scale_output 