import numpy as np
import random

from tqdm import tqdm

import argparse
import yaml

from functools import partial
from scipy import special as ss

from einops import rearrange, repeat
import opt_einsum as oe

import tensorflow as tf
from tensorflow import keras as ks
ls = ks.layers
import yaml

import argparse
from argparse import ArgumentParser
import einops
import keras_nlp

contract = oe.contract
contract_expression = oe.contract_expression

from util_layers import pytorch_he_uniform_init, Conv1d_with_init
from models.S4Model import S4Layer

import tensorflow_models as tfm


''' Standalone CSDI + S4Model imputer. The notebook contains CSDI and S4 functions and utilities. 
However the imputer is located in the last class of the notebook, please see more documentation of use there.'''
log = tf.get_logger()

class CSDIS4(ls.Layer):
    def __init__(self,
                    target_dim, timeemb, featureemb, 
                    num_steps,
                    channels,
                    diffusion_embedding_dim,
                    layers,
                    nheads,
                    s4_lmax=100,
                    conv_channels_first=True,
                    time_layer_d_state=64,
                    # eval_all_timestep=False,
                    **kwargs):

        super().__init__()

        
        self.target_dim = target_dim
        self.emb_time_dim = timeemb
        self.emb_feature_dim = featureemb
        self.num_steps = num_steps
        self.layers = layers
        self.nheads = nheads
        self.s4_lmax = s4_lmax
        
        # self.eval_all_timestep = eval_all_timestep
        
        self.input_dim = 2 
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1 #+1 for conditional mask

        # self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)
        self.embed_layer = ls.Embedding(input_dim=self.target_dim, 
                                output_dim=self.emb_feature_dim,
                                embeddings_initializer='normal')

        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.channels = channels
        
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=self.num_steps,
            embedding_dim=self.diffusion_embedding_dim)

        self.residual_layers = [
                ResidualBlock(
                    side_dim=self.emb_total_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=self.diffusion_embedding_dim,
                    nheads=self.nheads,
                    s4_lmax = self.s4_lmax,
                    conv_channels_first=conv_channels_first,
                    time_layer_d_state=time_layer_d_state)
                    for _ in range(self.layers)
            ]

        self.input_projection = Conv1d_with_init(self.input_dim, self.channels, 1, conv_channels_first=conv_channels_first)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1, conv_channels_first=conv_channels_first)
        
        # Output of model should be float32, whether or not mixed precision is used
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1, kernel_initializer=tf.keras.initializers.Zeros(),conv_channels_first=conv_channels_first, dtype=tf.float32)

        

    def build(self, input_shape):
        
        super().build(input_shape)
               
        self.built =True

    def call(self, x, cond_mask, diffusion_step, observed_timesteps=None):
        
        B, inputdim, K, L = x.shape

        if observed_timesteps is None:
            observed_timesteps = tf.tile( tf.range(0,L)[tf.newaxis], (B, 1))

        # This is 
        diffusion_step_emb = self.diffusion_embedding( diffusion_step )
        cond_mask = tf.cast(cond_mask, self.compute_dtype)
        cond_emb = self.get_side_emb( observed_timesteps, cond_mask ) #(B,L,H,emb)
        
        x = tf.reshape(x, (B, inputdim, K*L) )
        # x = tf.transpose(x, (0,2,1))
        x = self.input_projection( x )
        x = tf.nn.relu(x)
        x = tf.reshape(x, (B, self.channels, K, L) )

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_emb, diffusion_step_emb)
            skip.append(skip_connection)

        x = tf.reduce_sum( tf.stack(skip), axis=0) / tf.math.sqrt( tf.cast( len(self.residual_layers), dtype=x.dtype) ) 
            

        x = tf.reshape(x, (B, self.channels, K * L) )
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = tf.nn.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = tf.reshape(x,(B, K, L))
        return x

    def get_side_emb(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.broadcast_to( time_embed[:,:,tf.newaxis, :], (B, L, K, self.emb_time_dim ) )
        
        feature_embed = self.embed_layer(tf.range(self.target_dim))  # (K,emb)
        feature_embed = tf.broadcast_to( 
                            feature_embed[tf.newaxis, tf.newaxis, ...],
                            (B, L, *feature_embed.shape) )

        side_info = tf.concat([time_embed, feature_embed], axis=-1)  # (B,L,K,*)
        # side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        side_info = tf.transpose( side_info, perm=(0, 3, 2, 1) )

        
        side_mask = cond_mask[:, tf.newaxis, ...]  # (B,1,K,L)
        side_info = tf.concat([side_info, side_mask], axis=1)   

        return side_info

    def time_embedding(self, pos, d_model=128):
        # pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        pe = np.zeros( (pos.shape[0], pos.shape[1], d_model) )
        position = pos[:,:, np.newaxis,...]
        
        # div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        div_term = 1 / np.power(10000.0, np.arange(0, d_model, 2) / d_model)

        pe[:, :, 0::2] = np.sin(position * div_term)
        pe[:, :, 1::2] = np.cos(position * div_term)

        pe = tf.constant(pe, self.compute_dtype)
        return pe
    
    @staticmethod
    def parse_config(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        
        #NOTE: param below should be renamed, technically it is acctually the input channel dimension, not target dimension

        parser.add_argument("--target_dim", default=12, type=int)

        parser.add_argument("--timeemb", default=16, type=int)
        parser.add_argument("--featureemb", default=128, type=int)
        parser.add_argument("--diffusion_embedding_dim", default=128, type=int)

        parser.add_argument("--channels", default=64, type=int)
        
        parser.add_argument("--layers", default=4, type=int)
        parser.add_argument("--nheads", default=8, type=int)
        parser.add_argument("--s4_lmax", default=100, type=int)

        parser.add_argument("--conv_channels_first", type= lambda x: bool(int(x)), default=True)

        parser.add_argument("--num_steps", type=int, default=50 )

        parser.add_argument("--time_layer_d_state", type=int, default=8)
        config_model = parser.parse_known_args()[0]
        

        return config_model

class DiffusionEmbedding(ls.Layer):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.num_steps = num_steps
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        
        self.projection1 = ls.Dense(projection_dim, 
                                kernel_initializer= pytorch_he_uniform_init(embedding_dim / 2),
                                bias_initializer= pytorch_he_uniform_init(embedding_dim / 2))

        self.projection2 = ls.Dense(projection_dim, 
                                kernel_initializer= pytorch_he_uniform_init(projection_dim / 2),
                                bias_initializer= pytorch_he_uniform_init(projection_dim / 2))

    def build(self, input_shape):
        super().build(input_shape)
        self.embedding = tf.constant(self._build_embedding(self.num_steps, self.embedding_dim / 2))
        self.built = True


    def _build_embedding(self, num_steps, dim=64):
        steps = tf.range(num_steps, dtype=tf.float32)[:,tf.newaxis]  # (T,1)
        frequencies = 10.0 ** (tf.range(dim) / (dim - 1) * 4.0)[tf.newaxis, ...]  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        return table
    
    def call(self, diffusion_step):
        x = tf.gather( self.embedding, diffusion_step) # b, 1, 1 ,d
        x = self.projection1(x)
        x = ks.activations.swish(x)
        x = self.projection2(x)
        x = ks.activations.swish(x)
        return x # b, 1, 1, d

class ResidualBlock(ls.Layer):
    
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, s4_lmax=100, conv_channels_first=True, time_layer_d_state=64):
        super().__init__()
        
        self.diffusion_projection = ls.Dense( channels,
                                kernel_initializer=pytorch_he_uniform_init(diffusion_embedding_dim),
                                bias_initializer=pytorch_he_uniform_init(diffusion_embedding_dim))

        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1, conv_channels_first=conv_channels_first)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1,conv_channels_first=conv_channels_first)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1,conv_channels_first=conv_channels_first)

        self.time_layer = S4Layer(features=channels, lmax=s4_lmax, d_state=time_layer_d_state )
        self.feature_layer = get_tf_trans(heads=nheads, layers=1, channels=channels)

                
    def call(self, x, cond_info, diffusion_emb):
        base_shape = x.shape
        B, channel, K, L = x.shape
        
        x = tf.reshape(x, (B, channel, K*L))

        diffusion_emb = self.diffusion_projection(diffusion_emb)[..., tf.newaxis]  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape) #(B, channel, ..)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = tf.reshape(cond_info, (B, cond_dim, K * L))
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = tf.split(y, 2, axis=1)
        y = tf.sigmoid(gate) * tf.tanh(filter)
        y = self.output_projection(y)

        residual, skip = tf.split(y,2,axis=1)
        x = tf.reshape(x, base_shape)
        residual = tf.reshape(residual, base_shape)
        skip = tf.reshape(skip, base_shape)
        return (x + residual) / tf.cast(tf.math.sqrt(2.0),self.compute_dtype), skip
    
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        # Original
        # y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        # y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        
        #Simple
        y = einops.rearrange( y, 'b c (k l) -> (b k) c l ',b=B, c=channel, k=K, l=L)        
        y = self.time_layer(y)  # B*K, C, L      
        y = einops.rearrange(y, '(b k) c l -> b c (k l)', b=B, k=K, c=channel, l=L)
        
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        # y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        
        y = einops.rearrange( y, 'b c (k l) -> (b l) k c',b=B, c=channel, k=K, l=L)        
        y = self.feature_layer(y)        
        y = einops.rearrange(y, '(b l) k c -> b c (k l)',k=K, b=B)
        
        return y

def get_tf_trans(heads=8, layers=1, channels=64):
    
    # encoder = keras_nlp.layers.TransformerEncoder(
    #     intermediate_dim = channels, 
    #     num_heads = heads,
    #     dropout = 0.0,
    #     activation="gelu",
    # )
    
    
    encoder = tfm.nlp.models.TransformerEncoder(
        num_layers=1,
        num_attention_heads=heads,
        
        intermediate_size=channels, #64
        
        activation="gelu",
        norm_first=False,
        norm_epsilon=1e-5
    )
    
    return encoder

# region ==== Imputation stuff e.g. loss/diffuse
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
    # mask to introduce =  0's to impute, 1's to preserve.
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    mask = mask.astype("float32")
    gt_masks = observed_masks * mask

    return observed_values, observed_masks, gt_masks

# endregion

# # region ==== Dataset 
# class Custom_Impute_Dataset(Dataset):
#     def __init__(self, series, mask, use_index_list=None, path_save=''):
#         self.series = series
#         self.n_channels = series.size(2)
#         self.length = series.size(1)
#         self.mask = mask 

#         self.observed_values = []
#         self.observed_masks = []
#         self.gt_masks = []
#         path = f"{path_save}data_to_impute_missing" + ".pk"

#         if not os.path.isfile(path):  # if datasetfile is none, create
#             for sample in series:
                
#                 sample = sample.detach().cpu().numpy()
#                 observed_masks = sample.copy()
#                 observed_masks[observed_masks!=0] = 1 
#                 gt_masks = mask
                
#                 #observed_values, observed_masks, gt_masks = mask_missing_impute(sample, mask)
                

                
#                 self.observed_values.append(sample)
#                 self.observed_masks.append(observed_masks)
#                 self.gt_masks.append(gt_masks)


#             #tmp_values = self.observed_values.reshape(-1, self.n_channels)
#             #tmp_masks = self.observed_masks.reshape(-1, self.n_channels)
#             #mean = np.zeros(self.n_channels)
#             #std = np.zeros(self.n_channels)
#             #for k in range(self.n_channels):
#             #    c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
#             #    mean[k] = c_data.mean()
#             #    std[k] = c_data.std()
#             #self.observed_values = ((self.observed_values - mean) / std * self.observed_masks)

#             #with open(path, "wb") as f:
#             #   pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)
#         #else:
#             #with open(path, "rb") as f:
#                 #self.observed_values, self.observed_masks, self.gt_masks = pickle.load(f)
        
#         if use_index_list is None:
#             self.use_index_list = np.arange(len(self.observed_values))
#         else:
#             self.use_index_list = use_index_list

#     def __getitem__(self, org_index):
#         index = self.use_index_list[org_index]
#         s = {
#             "observed_data": self.observed_values[index],
#             "observed_mask": self.observed_masks[index],
#             "gt_mask": self.gt_masks[index],
#             "timepoints": np.arange(self.length),
#         }
#         return s

#     def __len__(self):
#         return len(self.use_index_list)

# def get_dataloader_impute(series, mask, batch_size=4, len_dataset=100):
#     indlist = np.arange(len_dataset)
#     impute_dataset = Custom_Impute_Dataset(series=series, use_index_list=indlist,mask=mask)
#     impute_loader = DataLoader(impute_dataset, batch_size=batch_size, shuffle=False)

#     return impute_loader

## endregion 




    


