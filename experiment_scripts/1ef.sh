#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
# export LD_LIBRARY_PATH="/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/:/home/akann1warw1ck/virtual_envs/anaconda3/envs/tsrl/lib/python3.10/site-packages/tensorrt/"
export LD_LIBRARY_PATH="/home/akann1w0w1ck/miniconda3/lib/:/home/akann1w0w1ck/miniconda3/lib/python3.10/site-packages/tensorrt/"


# Part e)
# CSDIS4 - Return prediction pn holidays across whole dataset
CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1e_csdis4" \
 --window_sample_count "3" --window_size "20" --model_name "CSDIS4" \
 --dataset_name "stocks" --test_set_method "holidays_only" \
 --diffusion_method "csdi" --epochs "150" --val_metrics "mae_rmse" --pred_metrics "logreturn" \
  --pred_samples "2"  --mask_method "bm_channelgroup" --missing_k "0 28 29 102 103 123" --batch_size "8"\
 --batch_size_val "8"   --mixed_precision --conv_channels_first "0" --target_dim "126" \
 --num_steps "50" --s4_lmax "22" --layers "4" --channels "128" --featureemb "128" --timeemb "16" \
  --diffusion_embedding_dim "128" --learning_rate "1e-3" --weight_decay "1e-6"

# SSSDS4 - Return prediction   
CUDA_VISIBLE_DEVICES=$1 python3 -O train.py --exp_name "yahoostocks_1e_sssds4" 
--window_sample_count "3" --window_size "10" --model_name "SSSDS4" --dataset_name "stocks" \
 --diffusion_method "alvarez" --epochs "150" --val_metrics "mae_rmse"
 --test_set_method "holidays_only" --pred_metrics "logreturn" --pred_samples "2" 
 --mask_method "bm_channelgroup" --missing_k "0 28 29 102 103 123" --batch_size "8" 
 --batch_size_val "8" --num_res_layers "4"  --in_channels "126" --out_channels "126" \
  --mixed_precision --conv_channels_first "0" --T "100" --s4_lmax "22" --diffusion_step_embed_dim_mid "128"
  --diffusion_step_embed_dim_out "128" --diffusion_step_embed_dim_in "64" --res_channels "64" \
   --skip_channels "64" --learning_rate "1e-3" --dropout "0.4" --learning_rate "1e-5"
                
                                                

# Part f)
# CSDIS4 - Compare error on holiday prediction to error on bm prediction
CUDA_VISIBLE_DEVICES=$1 python3 -O train_f.py --exp_name "yahoostocks_1f_csdis4" --window_sample_count "3" --window_size "20" --model_name "CSDIS4" --dataset_name "stocks" --test_set_method "holidays_only" \
 --diffusion_method "csdi" --epochs "150" --val_metrics "mae_rmse" --pred_metrics "logreturn" --pred_samples "2"  --mask_method "bm_channelgroup" --missing_k "0 28 29 102 103 123" --batch_size "8" \
 --batch_size_val "8"   --mixed_precision --conv_channels_first "0" --target_dim "126" --start_prop_train "0.0" --end_prop_train "0.6" --start_prop_val "0.6" --end_prop_val "0.8" --start_prop_test "0.8" --end_prop_test "1.0" \
 --num_steps "50" --s4_lmax "22" --layers "4" --channels "128" --featureemb "128" --timeemb "16"  --diffusion_embedding_dim "128"   --learning_rate "1e-3" --weight_decay "1e-6"

# # SSSDS4 - Compare error on holiday prediction to error on bm prediction 
# CUDA_VISIBLE_DEVICES=$1 python3 -O train_f.py --exp_name "yahoostocks_1f_sssds4" --window_sample_count "10" --window_size "20" --model_name "SSSDS4" --dataset_name "stocks" \
#  --diffusion_method "alvarez" --epochs "150" --val_metrics "mae_rmse" --test_set_method "holidays_only" --pred_metrics "logreturn" --pred_samples "2"  --mask_method "bm_channelgroup" --missing_k "0 28 29 102 103 123" --batch_size "8" --batch_size_val "8" --num_res_layers "4" \
#  --in_channels "126" --out_channels "126" --mixed_precision --conv_channels_first "0" --T "100" --s4_lmax "22" --diffusion_step_embed_dim_mid "128" --diffusion_step_embed_dim_out "128" \
#  --diffusion_step_embed_dim_in "64" --res_channels "64" --skip_channels "64" --start_prop_train "0.0" --end_prop_train "0.6" --start_prop_val "0.6" --end_prop_val "0.8" --start_prop_test "0.8" --end_prop_test "1.0" --dropout "0.4"
