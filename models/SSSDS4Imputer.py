import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras as ks
ls = ks.layers
import tensorflow_addons as tfa

from models.S4Model import S4Layer
from util_layers import pytorch_he_uniform_init, Conv1d_with_init, swish, Conv

import argparse
import numpy as np 

class SSSDS4Imputer(ls.Layer):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 conv_channels_first,
                 **kwargs):
        super(SSSDS4Imputer, self).__init__()

        # self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        
        self.conv_channels_first = conv_channels_first

        self.init_conv = tf.keras.Sequential( [ Conv(in_channels, filters=res_channels, kernel_size=1, 
                                                        conv_channels_first=self.conv_channels_first), ls.ReLU() ] )
        
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm,
                                             conv_channels_first=self.conv_channels_first)
        
        self.final_conv = ks.Sequential( [ Conv(skip_channels, 
                                                filters=skip_channels, kernel_size=1,
                                                    conv_channels_first=self.conv_channels_first),
                                            ls.ReLU(),
                                            ZeroConv1d(skip_channels, out_channels, conv_channels_first=self.conv_channels_first, dtype=tf.float32)]       
                                        )

    def call(self, noise, conditional, cond_mask, diffusion_steps):
        
        diffusion_step_embed = self.calc_diffusion_step_embedding(diffusion_steps)

        #noise #(b, seq, d)
        conditional = conditional * cond_mask
        conditional = tf.concat([conditional, cond_mask], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer(x, conditional, diffusion_step_embed)
        y = self.final_conv(x)
        return y

    def calc_diffusion_step_embedding(self, diffusion_steps):


        """
        Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

        Parameters:
        diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                    diffusion steps for batch data
        diffusion_step_embed_dim_in (int, default=128):  
                                    dimensionality of the embedding space for discrete diffusion steps
        
        Returns:
        the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
        """

        assert self.diffusion_step_embed_dim_in % 2 == 0

        half_dim = self.diffusion_step_embed_dim_in // 2
        _embed = np.log(10000) / (half_dim - 1)
        
        _embed = tf.math.exp(tf.range(half_dim, dtype=_embed.dtype) * -_embed)

        _embed = tf.cast(diffusion_steps,_embed.dtype) * _embed
        diffusion_step_embed = tf.concat([tf.math.sin(_embed),tf.math.cos(_embed)], 1)

        return diffusion_step_embed
    
    @staticmethod
    def parse_config(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        parser.add_argument("--in_channels", default=12, type=int )
        parser.add_argument("--out_channels", default=12, type=int )
        parser.add_argument("--res_channels",   default=256, type=int)
        parser.add_argument("--skip_channels", default=256, type=int )
        
        parser.add_argument("--num_res_layers", default=36, type=int) 


        parser.add_argument("--diffusion_step_embed_dim_in", default=128, type=int) #default=128)
        parser.add_argument("--diffusion_step_embed_dim_mid", default=256, type=int) #default=512)
        parser.add_argument("--diffusion_step_embed_dim_out", default=256, type=int) # default=512)

        parser.add_argument("--s4_lmax", default=100, type=int)
        parser.add_argument("--s4_d_state", default=64, type=int) #default=64)
        parser.add_argument("--s4_dropout", default=0.0, type=float)
        parser.add_argument("--s4_bidirectional", type = lambda x: bool(int(x)) , default=True)
        parser.add_argument("--s4_layernorm", default=1)

        parser.add_argument("--conv_channels_first", type=lambda x: bool(int(x)), default=False, help="Pass 1 for True, 0 for False. On windows CPU channels first is not functional. As such convolutional layers must have channels last") 

        config_model = parser.parse_known_args()[0]
        

        return config_model

class Residual_group(ls.Layer):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 conv_channels_first):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        
        self.fc_t1 = ls.Dense(diffusion_step_embed_dim_mid,
                                kernel_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_in),
                                bias_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_in )
        )

        self.fc_t2 = ls.Dense(diffusion_step_embed_dim_out, 
                                kernel_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_mid),
                                bias_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_mid )
        )

        
        self.residual_blocks = []

        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm,
                                                       conv_channels_first=conv_channels_first))
    
    def call(self, noise, conditional, diffusion_step_embed):

        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        skip = tf.zeros( (1,), dtype=self.compute_dtype )
        
        for n in tf.range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n](h, conditional, diffusion_step_embed) 
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  

class Residual_block(ls.Layer):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 conv_channels_first):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.conv_channels_first = conv_channels_first
        self.skip_channels = skip_channels

        # self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        self.fc_t = ls.Dense(self.res_channels, 
                                kernel_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_out),
                                bias_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_out)
        )

        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          d_state=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
 
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, conv_channels_first=conv_channels_first)

        self.S42 = S4Layer(features=2*self.res_channels, 
                            lmax=s4_lmax,
                            d_state=s4_d_state,
                            dropout=s4_dropout,
                            bidirectional=s4_bidirectional,
                            layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1, conv_channels_first=conv_channels_first)  

        self.res_conv = Conv1d_with_init( res_channels, res_channels, kernel_size=1, weight_norm=True, conv_channels_first=conv_channels_first )

        self.skip_conv = Conv1d_with_init( res_channels, skip_channels, kernel_size=1, weight_norm=True, conv_channels_first=conv_channels_first )

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, x, cond, diffusion_step_embed ):
        h = x # (b, c, l)
        B, C, L = x.shape 
        # assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        # part_t = part_t.view([B, self.res_channels, 1])  
        part_t = part_t[:,:,tf.newaxis] # (b, c , 1)
        h = h + part_t
        
        h = self.conv_layer(h) # (b, c2, l)
        h = self.S41(h)
        
        
        # assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        
        # h = self.S42(h.permute(2,0,1)).permute(1,2,0)
        # h = self.S42(h)
        h = self.S42(h)

        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out) #this weight gets no grsds
        # assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * tf.cast( tf.math.sqrt(0.5), x.dtype), skip  # normalize for training stability

class ZeroConv1d(ls.Layer):
    def __init__(self, in_channels, filters, conv_channels_first=False, dtype=None):
        super(ZeroConv1d, self).__init__()
        
        self.conv_channels_first = conv_channels_first
        self.conv = ls.Conv1D(filters, kernel_size=1, padding='valid', 
                        data_format='channels_first' if conv_channels_first else 'channels_last', 
                        kernel_initializer=tf.keras.initializers.Zeros(),
                        bias_initializer=tf.keras.initializers.Zeros(),
                        dtype=dtype)

        if conv_channels_first==False:
            # This layer always recieves input:(b, c, seq)
            self.conv = tf.keras.Sequential(
                [   
                    ls.Permute((2,1), dtype=dtype),
                    self.conv,
                    ls.Permute((2,1), dtype=dtype),
                ]
            )
            
    def call(self, x):
        out = self.conv(x)
        return out
    
    def build(self, input_shape):
        if self.conv_channels_first == False:
            self.conv.layers[0].build(input_shape)
            self.conv.layers[2].build(input_shape)
            self.conv.layers[1].build(tf.TensorShape([input_shape[0],input_shape[2], input_shape[1]]))
        else:
    
            self.conv.build(input_shape)
        
        self.conv.built = True


