import numpy as np
import random
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import datetime
import json
import yaml
import os
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers as ls
import tensorflow_probability as tfp
import math
from util_layers import Conv1d_with_init, pytorch_he_uniform_init
''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''


    
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    # return 2 * torch.sum(torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))
    return 2 * tf.reduce_sum(tf.math.abs((forecast - target) * eval_points * (tf.cast(target <= forecast, tf.float32) * 1.0 - q)))

def calc_denominator(target, eval_points):
    # return torch.sum(torch.abs(target * eval_points))
    return tf.reduce_sum(tf.math.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            # q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
            q_pred.append(tf.convert_to_tensor(np.quantile(forecast[j: j + 1], quantiles[i], axis=1)))
        q_pred = tf.concat(q_pred, axis=0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
        
    # return CRPS.item() / len(quantiles)
    return CRPS.numpy().item() / len(quantiles)

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

# def Conv1d_with_init(in_channels, out_channels, kernel_size):
#     # layer = nn.Conv1d(in_channels, out_channels, kernel_size)
#     # nn.init.kaiming_normal_(layer.weight)

#     layer = ls.Conv1D(out_channels, kernel_size, padding='same', data_format='channels_first', 
#         kernel_initializer=tf.keras.initializers.HeNormal(),
#         bias_initializer=pytorch_he_uniform_init( in_channels*kernel_size)
#         )

#     return layer


class CSDI_base(ls.Layer):
    def __init__(self, target_dim, config, device):
        super().__init__()
        # self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        
        #TODO: ensure that initalization below is equivalent to pytorch init
        # self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)
        self.embed_layer = ls.Embedding(self.target_dim, self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        # self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        self.alpha_torch = tf.constant(self.alpha, tf.float32)[:, tf.newaxis, tf.newaxis, ...]

    def time_embedding(self, pos, d_model=128):
        pe = tf.zeros( (pos.shape[0], pos.shape[1], d_model) )
        position = tf.expand_dims(pos, 2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        div_term = 1 / torch.math.pow(10000.0, tf.range(0, d_model, 2) / d_model)
        # pe[:, :, 0::2] = torch.sin(position * div_term)
        # pe[:, :, 1::2] = torch.cos(position * div_term)
        pe[:, :, 0::2] = tf.math.sin(position * div_term)
        pe[:, :, 1::2] = tf.math.cos(position * div_term)
        return pe


    def get_randmask(self, observed_mask):
        # This function applies a random mask to a random proportion of the observed mask
            # the masking is independent across each channel, but the same proportion across channels

        # Logic: 
         # - calculate number of elements to mask as propotion of values True in observed mask
         # - create an array of floats between [0,1] same shape as observed mask
         # - set the top_k floats in this array, where k is the prop of elements we want to mask, to -1
         # - then create bool array by >0, reshape to observed_mask original shape, then cast back to float32
         # this is the conditional mask

        # rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        # rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        
        rand_for_mask = tf.random.uniform(observed_mask.shape) * observed_mask
        rand_for_mask = tf.reshape(rand_for_mask, (rand_for_mask.shape[0], -1))

        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  
            # num_observed = observed_mask[i].sum().item()
            num_observed = tf.reduce_sum( observed_mask[i] ).numpy().item()
            num_masked = round(num_observed * sample_ratio)

            # rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            rand_for_mask[i][tf.math.top_k(rand_for_mask[i], num_masked).indices] = -1

        # cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        cond_mask = tf.cast( tf.reshape(rand_for_mask > 0, observed_mask.shape ), tf.float32)

        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        # if target_strategy == "mix" - then w/ 50% chance return randmask output, 50% chance return the observed mask

        for_pattern_mask = for_pattern_mask if for_pattern_mask else observed_mask

        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = tf.identity(observed_mask)

        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask


    def get_side_info(self, observed_tp, cond_mask):
        
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.tile( tf.unsqueeze( time_embed, 2), (-1, -1, K, -1))
        
        feature_embed = self.embed_layer(tf.range(self.target_dim,dtype=tf.float32))  # (K,emb)
        feature_embed = tf.tile(feature_embed[tf.newaxis, tf.newaxis, ...], (B, L, 1, 1))

        side_info = tf.concat((time_embed, feature_embed), axis=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask[:, tf.newaxis, ...]  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], axis=1)

        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            # loss_sum += loss.detach()
            loss_sum += loss
            
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            # t = (torch.ones(B) * set_t).long().to(self.device)
            t = tf.cast(tf.ones(B)*set_t, tf.int64)
        else:
            # t = torch.randint(0, self.num_steps, [B]).to(self.device)
            t = tf.experimental.numpy.random.randint( 0, self.num_steps, (B) )

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        # noise = torch.randn_like(observed_data).to(self.device)
        noise = tf.random.normal(observed_data.shape)
        # noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask

        residual = (noise - predicted) * target_mask
        
        num_eval = tf.reduce_sum(target_mask)
        loss = tf.reduce_sum(residual ** 2) / (num_eval if num_eval > 0 else 1)
        
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data[:, tf.newaxis, ...]  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data)[:, tf.newaxis, ...]
            noisy_target = ((1 - cond_mask) * noisy_data)[:, tf.newaxis, ...]
            total_input = tf.concat((cond_obs, noisy_target), axis=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        # imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        imputed_samples = tf.zeros( (B, n_samples, K, L))

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    # noise = torch.randn_like(noisy_obs)
                    noise = tf.random.normal(noisy_obs.shape)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            # current_sample = torch.randn_like(observed_data)
            current_sample = tf.random.normal(observed_data.shape )

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input[:,tf.newaxis,...]  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data)[:,tf.newaxis,...]
                    noisy_target = ((1 - cond_mask) * current_sample)[:,tf.newaxis,...]
                    diff_input = tf.concat((cond_obs, noisy_target), axis=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, tf.constant([t]))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    # noise = torch.randn_like(current_sample)
                    noise = tf.random.normal(current_sample.shape)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample

        imputed_samples = tf.stop_gradient(imputed_samples)
        return imputed_samples

    def call(self, batch, is_train=1):
        (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,_) = self.process_data(batch)
        # for_pattern_mask == observed_mask

        #TODO: {rilwan.adewoyin} need to make sure masks align to masking system in old code
        #NOTE: what is the difference between gt_mask and observed_mask
        #NOTE: what is the diff between target strategy and masking
        # original logic
        # precursor - In CSDI the mask is created in the dataset class using masking types of 'rm','nrm' and 'bm'
            # observed masks is just where observed data is nan
            # gt is essentially target masks, so observed_masks + any missingness ratio related mask
                # gt mask also reflects the rm,nrm,bm,tf
            # target strategy is how you select the targets, masking strategy is how you select which 
        # if imputing (predicting/testing):
            # use the mask created in the CSDIImputation Dataset
        # if training and target_strategy is not random:
            # ignore gt_mask

        # if is_train == 0:
        #     cond_mask = gt_mask
        # elif self.target_strategy != "random":
        #     cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        # else:
        #     cond_mask = self.get_randmask(observed_mask)

        if is_train == 0:
            # if not train, then use the mask specified 
            cond_mask = gt_mask
        
        #NOTE: CSDI uses method whereby

        #NOTE: this below should be part of the predict only - or part of the datasets method to create gt_mask
        # elif self.target_strategy in ["rm","nrm","bm","tf"] :
        #     #NOTE: this option is unifying the parameter space and implementation of CSDI and SSSDS4
        #     self.diffusion.get_mask( masking=self.target_strategy, observed_mask=observed_mask )

        elif self.target_strategy == "random":
            cond_mask = self.get_randmask(observed_mask)

        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
    
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (observed_data,observed_mask,observed_tp,gt_mask,_,cut_length) = self.process_data(batch)
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):  
                target_mask[i, ..., 0: cut_length[i].item()] = 0
                
        return samples, observed_data, target_mask, observed_mask, observed_tp


class DiffusionEmbedding(ls.Layer):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        # self.register_buffer(
        #     "embedding",
        #     self._build_embedding(num_steps, embedding_dim / 2),
        #     persistent=False)
        # self.projection1 = nn.Linear(embedding_dim, projection_dim)
        # self.projection2 = nn.Linear(projection_dim, projection_dim)

        #NOTE: pytorch and keras implementation of 'he_uniform' are not identical
        
        self.embedding = tf.constant(self._build_embedding(num_steps, embedding_dim / 2))
        self.projection1 = ls.Dense(projection_dim, 
                                kernel_initializer= pytorch_he_uniform_init(embedding_dim / 2),
                                bias_initializer= pytorch_he_uniform_init(embedding_dim / 2))

        self.projection2 = ls.Dense(projection_dim, 
                                kernel_initializer= pytorch_he_uniform_init(embedding_dim / 2),
                                bias_initializer= pytorch_he_uniform_init(embedding_dim / 2))

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        # x = F.silu(x)
        x = ks.activations.swish(x)
        x = self.projection2(x)
        # x = F.silu(x)
        x = ks.activations.swish(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        # steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        steps = tf.range(num_steps)[:,tf.newaxis]  # (T,1)
        # frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        frequencies = 10.0 ** (tf.range(dim) / (dim - 1) * 4.0)[tf.newaxis, ...]  # (1,dim)
        table = steps * frequencies  # (T,dim)
        # table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        # nn.init.zeros_(self.output_projection2.weight)
        self.output_projection2 =  ls.Conv1D(self.channels, 1, padding='same', data_format='channels_first', 
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=pytorch_he_uniform_init( self.channels*1)
            )
        self.residual_layers = [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]

    def call(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.get_shape()

        x = x.reshape(B, inputdim, K * L)
        x = tf.reshape(x, (B, inputdim, K * L) )
        x = self.input_projection(x)
        x = ks.activations.relu(x)
        x = tf.reshape( x, (B, self.channels, K, L) )

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        # x = x.reshape(B, self.channels, K * L)
        x = tf.reduce_sum(tf.stack(skip), axis=0) / tf.math.sqrt(len(self.residual_layers))
        x = tf.reshape(x,(B, self.channels, K * L))
        x = self.output_projection1(x)  # (B,channel,K*L)
        # x = F.relu(x)
        x = ks.activations.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = tf.reshape(x,(B, K, L))
        return x

    
class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        # self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.diffusion_projection = ls.Dense(channels, 
                                        kernel_initializer= pytorch_he_uniform_init( diffusion_embedding_dim), 
                                        bias_initializer= pytorch_he_uniform_init( diffusion_embedding_dim))
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        # y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        # y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        
        c = channel
        y = tf.reshape(tf.transpose(tf.reshape(y,(B, c, K, L)), (0, 2, 1, 3)), (B * K, c, L))
        y = tf.transpose(self.time_layer(tf.transpose(y, (0, 2, 1))), (0, 1 , 2)) ### B*K, L, channel
        y = tf.reshape(tf.transpose(tf.reshape(y, (B, K, c, L)), (0, 2, 1, 3)), (B, c, K * L))
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        # y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        
        c = channel
        y = tf.reshape( tf.transpose( tf.reshape(y, (B, c, K, L)), (0, 3, 1, 2) ), (B * L, c, K) )
        y = tf.transpose(self.feature_layer(tf.transpose(y, (0, 2, 1))), (0, 1 ,2)) ### B*L, K, channel
        y = tf.reshape(tf.transpose(tf.reshape(y, (B, L, c, K)), (0, 2, 3, 1)), (B, c, K * L))

        return y

    def call(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.get_shape()
        base_shape = x.get_shape()
        x = tf.reshape(x, (B, channel, K * L) )

        diffusion_emb = self.diffusion_projection(diffusion_emb)[tf.newaxis, ...]  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        
        
        _, cond_dim, _, _ = cond_info.get_shape()
        cond_info = tf.reshape(cond_info, (B, cond_dim, K * L))
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        
        # gate, filter = torch.chunk(y, 2, dim=1)
        gate, filter = tf.split(y, 2, axis=1)
        # y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        # residual, skip = torch.chunk(y, 2, dim=1)
        gate, filter = tf.split(y, 2, axis=1)
        x = tf.reshape(x,base_shape)
        residual = tf.reshape(residual, base_shape)
        skip = tf.reshape(skip, base_shape)
        return (x + residual) / tf.math.sqrt(2.0), skip


    
class CSDI_Custom(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Custom, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # observed_data = batch["observed_data"].to(self.device).float()
        # observed_mask = batch["observed_mask"].to(self.device).float()
        # observed_tp = batch["timepoints"].to(self.device).float()
        # gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = tf.cast(batch["observed_data"], tf.float32)
        observed_mask = tf.cast(batch["observed_mask"], tf.float32)
        observed_tp =   tf.cast(batch["timepoints"], tf.float32)
        gt_mask =       tf.cast(batch["gt_mask"], tf.float32)

        # observed_data = observed_data.permute(0, 2, 1)
        # observed_mask = observed_mask.permute(0, 2, 1)
        # gt_mask = gt_mask.permute(0, 2, 1)

        observed_data = tf.transpose(observed_data, (0, 2, 1))
        observed_mask = tf.transpose(observed_mask, (0, 2, 1))
        gt_mask =       tf.transpose(gt_mask, (0, 2, 1))

        # cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        # for_pattern_mask = observed_mask

        cut_length = tf.cast(tf.zeros(len(observed_data)), tf.int64)
        for_pattern_mask = observed_mask

        return (observed_data, observed_mask, observed_tp,
                gt_mask, for_pattern_mask, cut_length)
    
    
def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_nrm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)

    for channel in range(gt_masks.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_bm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)
    s_nan = random.choice(list_of_segments_index)

    for channel in range(gt_masks.shape[1]):
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_impute(data, mask):
    
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    mask = mask.astype("float32")
    gt_masks = observed_masks * mask

    return observed_values, observed_masks, gt_masks

#TODO: rename this class away - remove 'train'
class Custom_Train_Dataset_CSDI():
    def __init__(self, series, path_save=None, use_index_list=None, missing_ratio_or_k=0.0, masking='rm', ms=None):
        self.series = series
        self.length = series.shape[1]
        self.n_channels = series.shape[2]

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = f"{path_save}data_train_random_points" + str(missing_ratio_or_k) + ".pk" if path_save else False

        if not os.path.isfile(path):  # if datasetfile is none, create
            for sample in series:
                if masking == 'rm':
                    sample = sample
                    observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, missing_ratio_or_k)

                elif masking == 'nrm':
                    sample = sample
                    observed_values, observed_masks, gt_masks = mask_missing_train_nrm(sample, missing_ratio_or_k)

                elif masking == 'bm':
                    sample = sample
                    observed_values, observed_masks, gt_masks = mask_missing_train_bm(sample, missing_ratio_or_k)

                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                        
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s
    
    def get_dloader(self, batch_size, shuffle_buffer_size=0, prefetch=False, cache=False):
        index = self.use_index_list
        data = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": tf.constant(np.arange(len(index))),
        }

        dset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle_buffer_size>0:
            dset = dset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        if prefetch:
            dset = dset.prefetch(tf.data.AUTOTUNE)

        dset = dset.batch(batch_size)

        return dset

    def __len__(self):
        return len(self.use_index_list)

    
class Custom_Impute_Dataset(Dataset):
    def __init__(self, series, mask, use_index_list=None, path_save=''):
        self.series = series
        self.n_channels = series.shape[2]
        self.length = series.shape[1]
        self.mask = mask 

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        path = f"{path_save}data_to_impute_missing" + ".pk" if path_save else False

        if not os.path.isfile(path):  # if datasetfile is none, create
            for sample in series:
                
                sample = sample #.detach().cpu().numpy()
                
                #TODO: deteermine which of the below ways to calc observed masks is best, 2nd is copied 
                observed_masks = sample
                observed_masks[observed_masks!=0] = 1 

                # observed_masks = np.ones(sample.shape)

                gt_masks = mask
                
                #observed_values, observed_masks, gt_masks = mask_missing_impute(sample, mask)
                
                self.observed_values.append(sample)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)

                
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s

    def get_dset(self, batch_size, shuffle_buffer_size=0, prefetch=False, cache=False):
        index = self.use_index_list
        data = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": tf.constant(np.arange(len(index))),
        }

        dset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle_buffer_size>0:
            dset = dset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        if prefetch:
            dset = dset.prefetch(tf.data.AUTOTUNE)

        dset = dset.batch(batch_size)

        return dset

    def __len__(self):
        return len(self.use_index_list)
    
    
def get_dataloader_train_impute(series,
                                batch_size=4,
                                missing_ratio_or_k=0.1,
                                train_split=0.7,
                                valid_split=0.9,
                                len_dataset=100,
                                masking='rm',
                               path_save='',
                               ms=None):
    indlist = np.arange(len_dataset)

    tr_i, v_i, te_i = np.split(indlist,
                               [int(len(indlist) * train_split),
                                int(len(indlist) * (train_split + valid_split))])

    dset_train = Custom_Train_Dataset_CSDI(series=series, use_index_list=tr_i,
                                         missing_ratio_or_k=missing_ratio_or_k, 
                                         masking=masking, path_save=None, ms=1)
    # train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    dloader_train = dset_train.get_dloader(batch_size, 
                        shuffler_buffer_size=len(dset_train)//2,
                        prefetch=True)
    dset_val = Custom_Train_Dataset_CSDI(series=series, use_index_list=v_i, 
                                         missing_ratio_or_k=missing_ratio_or_k, 
                                         masking=masking, path_save=None)    
    # valid_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=True)
    dloader_val = dset_train.get_dloader(batch_size, 
                        shuffler_buffer_size=0,
                        prefetch=True)

    dset_test = Custom_Train_Dataset_CSDI(series=series, use_index_list=te_i, 
                                        missing_ratio_or_k=missing_ratio_or_k, 
                                        masking=masking, path_save=None)
    # test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=True)
    dloader_test = dset_train.get_dloader(batch_size, 
                        shuffler_buffer_size=0,
                        prefetch=True)

    return dloader_train, dloader_val, dloader_test


def get_dataloader_impute(series, mask, batch_size=4, len_dataset=100):
    indlist = np.arange(len_dataset)
    dset_impute = Custom_Impute_Dataset(series=series, use_index_list=indlist, mask=mask)
    dloader_impute = dset_impute.get_dloader(batch_size, 
                        shuffler_buffer_size=0,
                        prefetch=True)

    return dloader_impute


# This is the trainer
class CSDIImputer(tf.keras.Model):
    def __init__(self):
        super(CSDIImputer,self).__init__()
        np.random.seed(0)
        random.seed(0)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = None
        
        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              masking ='rm',
              missing_ratio_or_k = 0.0,
              train_split = 0.7,
              valid_split = 0.2,
              epochs = 200,
              samples_generate = 10,
              path_save = "",
              batch_size = 16,
              lr = 1.0e-3,
              layers = 4,
              channels = 64,
              nheads = 8,
              difussion_embedding_dim = 128,
              beta_start = 0.0001,
              beta_end = 0.5,
              num_steps = 50,
              schedule = 'quad',
              is_unconditional = 0,
              timeemb = 128,
              featureemb = 16,
              target_strategy = 'random',  #random, mix
             ):
        
        '''
        CSDI training function. 
       
       
        Requiered parameters
        -series: Assumes series of shape (Samples, Length, Channels).
        -masking: 'rm': random missing, 'nrm': non-random missing, 'bm': black-out missing.
        -missing_ratio_or_k: missing ratio 0 to 1 for 'rm' masking and k segments for 'nrm' and 'bm'.
        -path_save: full path where to save model weights, configuration file, and means and std devs for de-standardization in inference.
        
        Default parameters
        -train_split: 0 to 1 representing the percentage of train set from whole data.
        -valid_split: 0 to 1. Is an adition to train split where 1 - train_split - valid_split = test_split (implicit in method).
        -epochs: number of epochs to train.
        -samples_generate: number of samples to be generated.
        -batch_size: batch size in training.
        -lr: learning rate.
        -layers: difussion layers.
        -channels: number of difussion channels.
        -nheads: number of difussion 'heads'.
        -difussion_embedding_dim: difussion embedding dimmensions. 
        -beta_start: start noise rate.
        -beta_end: end noise rate.
        -num_steps: number of steps.
        -schedule: scheduler. 
        -is_unconditional: conditional or un-conditional imputation. Boolean.
        -timeemb: temporal embedding dimmensions.
        -featureemb: feature embedding dimmensions.
        -target_strategy: strategy of masking. 
        -wandbiases_project: weight and biases project.
        -wandbiases_experiment: weight and biases experiment or run.
        -wandbiases_entity: weight and biases entity. 
        '''
       
        config = {}
        
        config['train'] = {}
        config['train']['epochs'] = epochs
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = lr
        config['train']['train_split'] = train_split
        config['train']['valid_split'] = valid_split
        config['train']['path_save'] = path_save
        
       
        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule
        
        config['model'] = {} 
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = masking
        
        print(json.dumps(config, indent=4))

        config_filename = path_save + "config_csdi_training"
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(config, f, indent=4)


        train_loader, valid_loader, test_loader = get_dataloader_train_impute(
            series=series,
            train_split=config["train"]["train_split"],
            valid_split=config["train"]["valid_split"],
            len_dataset=series.shape[0],
            batch_size=config["train"]["batch_size"],
            missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
            masking=config['model']['masking'],
            path_save=config['train']['path_save'])

        model = CSDI_Custom(config, self.device, target_dim=series.shape[2]) #.to(self.device)

        train(model=model,
              config=config["train"],
              train_loader=train_loader,
              valid_loader=valid_loader,
              path_save=config['train']['path_save'])

        evaluate(model=model,
                 test_loader=test_loader,
                 nsample=samples_generate,
                 scaler=1,
                 path_save=config['train']['path_save'])
        
    # def load_weights(self, 
    #                  path_load_model='',
    #                  path_config=''):
        
    #     self.path_load_model_dic = path_load_model
    #     self.path_config = path_config
    
    
    #     '''
    #     Load weights and configuration file for inference.
        
    #     path_load_model: load model weights
    #     path_config: load configuration file
    #     '''
    
    def impute(self,
               sample,
               mask,
               n_samples = 50,
               ):
        '''
        Imputation function 
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain. 
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        
        # if len(sample.shape) == 2:
        #     self.series_impute = torch.from_numpy(np.expand_dims(sample, axis=0))
        # elif len(sample.shape) == 3:
        #     self.series_impute = sample

        if len(sample.shape) == 2:
            self.series_impute = tf.constant(np.expand_dims(sample, axis=0))
        elif len(sample.shape) == 3:
            self.series_impute = sample

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.path_config, "r") as f:
            config = json.load(f)

        test_loader = get_dataloader_impute(series=self.series_impute,
                                            len_dataset=len(self.series_impute),
                                            mask=mask,
                                            batch_size=config['train']['batch_size'])

        model = CSDI_Custom(config, self.device, target_dim=self.series_impute.shape[2]) #.to(self.device)

        # model.load_state_dict(torch.load((self.path_load_model_dic)))

        model.load_weights(self.path_load_model_dic)
        
        imputations = evaluate(model=model,
                                test_loader=test_loader,
                                nsample=n_samples,
                                scaler=1,
                                path_save='')
        
        indx_imputation = ~mask.astype(bool)
            
        original_sample_replaced =[]
        
        for original_sample, single_n_samples in zip(sample.numpy(), imputations): # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]            
            single_sample_replaced = []
            for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
                sample_out = original_sample.copy()                         
                sample_out[indx_imputation] = sample_generated[indx_imputation].numpy()
                single_sample_replaced.append(sample_out)
            original_sample_replaced.append(single_sample_replaced)
            
        output = np.asarray(original_sample_replaced)
        
        
        return output

# This is called by CSDIImputer.train
def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=50, path_save=""):
    # optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    
    output_path = f"{path_save}model.pth"
    output_path_best = f"{path_save}model_best.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    pcd_sched = ks.optimizers.schedules.PiecewiseConstantDecay( 
        boundaries = [p1-1, p2-1],
        values = [ config["lr"]*( config.get('gamma',0.1)**pow ) for pow in range(0,2+1)  ] 
        )
    optimizer = ks.optimizers.Adam(pcd_sched, weight_decay=1e-6)

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        # model.train()

        with tqdm(train_loader, mininterval=1, maxinterval=1) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                with tf.GradientTape() as tape:
                    loss = model(train_batch, training=True)
                
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                it.set_postfix(
                    ordered_dict={"avg_epoch_loss": avg_loss / batch_no,"epoch": epoch_no + 1},refresh=False)
                
        # torch.save(model.state_dict(), output_path)
        model.save_weights(output_path)


        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            
            avg_loss_valid = 0        
            with tqdm(valid_loader, mininterval=50, maxinterval=50) as it:
                for batch_no, valid_batch in enumerate(it, start=1):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.numpy().item()
                    it.set_postfix(ordered_dict={"valid_avg_epoch_loss":avg_loss_valid/batch_no,"epoch":epoch_no},refresh=False)
        
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print("\n best loss is updated to ",avg_loss_valid/batch_no,"at",epoch_no+1)
                output_path_best
            
            model.save_weights(output_path_best)
        # try:
        #     wandb.log({"loss_valid": avg_loss_valid / batch_no})
        # except:
        #   pass

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save=""):
    # with torch.no_grad():
    #     model.eval()
    
    mse_total = 0
    mae_total = 0
    evalpoints_total = 0

    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []
    with tqdm(test_loader, mininterval=1.0, maxinterval=1.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            output = model.evaluate(test_batch, nsample)

            samples, c_target, eval_points, observed_points, observed_time = output
            samples = tf.transpose( samples, (0, 1, 3, 2) ) # (B,nsample,L,K)
            c_target = tf.transpose( c_target, (0, 2, 1) ) # (B,L,K)
            eval_points = tf.transpose( eval_points, (0, 2, 1) )
            observed_points = tf.transpose( observed_points,(0, 2, 1) )

            # samples_median = samples.median(dim=1)
            samples_median = tfp.stats.percentile( samples, 50,  axis=tf.constant(1,dtype=tf.int32), preserve_gradients=False )
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            mse_current = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
            mae_current = (tf.math.abs((samples_median - c_target) * eval_points)) * scaler

            # mse_total += mse_current.sum().item()
            # mae_total += mae_current.sum().item()
            # evalpoints_total += eval_points.sum().item()

            mse_total += tf.reduce_sum(mse_current).numpy().item()
            mae_total += tf.reduce_sum(mae_current).numpy().item()
            evalpoints_total +=  tf.reduce_sum(eval_points).numpy().item()

            it.set_postfix(ordered_dict={
                    "rmse_total": math.sqrt(mse_total / evalpoints_total),
                    "mae_total": mae_total / evalpoints_total,
                    "batch_no": batch_no}, refresh=True)
            
        with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
            all_target = tf.concat(all_target, axis=0)
            all_evalpoint = tf.concat(all_evalpoint, axis=0)
            all_observed_point = tf.concat(all_observed_point, axis=0)
            all_observed_time = tf.concat(all_observed_time, axis=0)
            all_generated_samples = tf.concat(all_generated_samples, axis=0)

            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

        CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

        with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
            pickle.dump(
                [
                    math.sqrt(mse_total / evalpoints_total),
                    mae_total / evalpoints_total, 
                    CRPS
                ], 
                f)
            print("RMSE:", math.sqrt(mse_total / evalpoints_total))
            print("MAE:", mae_total / evalpoints_total)
            print("CRPS:", CRPS)


    return all_generated_samples.numpy()



