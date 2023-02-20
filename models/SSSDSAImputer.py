# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras as ks
ls = ks.layers
import tensorflow_addons as tfa

from einops import rearrange
from models.S4Model import S4, LinearActivation, TransposedLinear
from util_layers import pytorch_he_uniform_init, Conv1d_with_init, pytorch_he_normal_init, swish, Conv

from collections.abc import Iterable   # import 
import argparse 

class SSSDSAImputer(ls.Layer):
    def __init__(
        self,
        d_model=128, 
        n_layers=6, #tick
        pool=[2, 2], 
        expand=2, 
        ff=2, 
        glu=True,
        unet:bool=True,
        dropout=0.0,
        in_channels=1, #tick
        out_channels=1, #tick
        diffusion_step_embed_dim_in=128, #tick
        diffusion_step_embed_dim_mid=512,#tick
        diffusion_step_embed_dim_out=512,#tick
        label_embed_dim=128,
        label_embed_classes=71,
        bidirectional=True,
        s4_lmax=1,
        s4_d_state=64,
        s4_dropout=0.0,
        s4_bidirectional=True,
        conv_channels_first=True
    ):
        
        """
        SaShiMi model backbone. 

        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level. 
                We use 8 layers for our experiments, although we found that increasing layers even further generally 
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels. 
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
                We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models 
                such as diffusion models like DiffWave.
            glu: use gated linear unit in the S4 layers. Adds parameters and generally improves performance.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling. 
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__()
        self.d_model = H = d_model
        self.unet = unet
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.conv_channels_first = conv_channels_first
        
        def s4_block(dim, stride):
          
            layer = S4(
                d_model=dim, 
                l_max=s4_lmax,
                d_state=s4_d_state,
                bidirectional=s4_bidirectional,
                postact='glu' if glu else None,
                dropout=dropout,
                transposed=True,
                #hurwitz=True, # use the Hurwitz parameterization for stability
                #tie_state=True, # tie SSM parameters across d_state in the S4 layer
                trainable={
                    'dt': True,
                    'A': True,
                    'P': True,
                    'B': True,
                }, # train all internal S4 parameters
                    
            )
            
                
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels = in_channels,
                label_embed_dim = label_embed_dim,
                stride=stride ,
                conv_channels_first=self.conv_channels_first    
            )

        def ff_block(dim, stride):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels = in_channels,
                label_embed_dim = label_embed_dim,
                stride=stride,
                conv_channels_first=self.conv_channels_first
            )

        # Down blocks
        d_layers = []
        for i, p in enumerate(pool):
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    if i == 0:
                        d_layers.append(s4_block(H, 1))
                        if ff > 0: d_layers.append(ff_block(H, 1))
                    elif i == 1:
                        d_layers.append(s4_block(H, p))
                        if ff > 0: d_layers.append(ff_block(H, p))
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand
        
        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H, pool[1]*2))
            if ff > 0: c_layers.append(ff_block(H, pool[1]*2))
        
        # Up blocks
        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, causal= not bidirectional))

            for _ in range(n_layers):
                if i == 0:
                    block.append(s4_block(H, pool[0]))
                    if ff > 0: block.append(ff_block(H, pool[0]))
                        
                elif i == 1:
                    block.append(s4_block(H, 1))
                    if ff > 0: block.append(ff_block(H, 1))

            u_layers.append(block)

        self.d_layers = d_layers 
        self.c_layers = c_layers
        self.u_layers = u_layers
        
        # self.norm = nn.LayerNorm(H)
        
        # l = len(H) if isinstance(H, Iterable ) else 1
        # axis= list( range(-1, -1-l, -1) )
        self.norm = ls.LayerNormalization( axis=1, epsilon=1e-5, center=True, scale=True) 

        self.init_conv=ks.Sequential(
            [Conv1d_with_init( in_channels,d_model,kernel_size=1, conv_channels_first=conv_channels_first),
            ls.Activation('relu')]
            )
        self.final_conv=ks.Sequential(
            [Conv1d_with_init( d_model, d_model,kernel_size=1,conv_channels_first=conv_channels_first),
            ls.Activation('relu'),
            Conv1d_with_init( d_model, out_channels, kernel_size=1,conv_channels_first=conv_channels_first, dtype=tf.float32)]
            )
        self.fc_t1 = ls.Dense(diffusion_step_embed_dim_mid,
                                kernel_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_in),
                                bias_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_in )
                    )
        self.fc_t2 = ls.Dense(diffusion_step_embed_dim_out, 
                                kernel_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_mid),
                                bias_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_mid )
        )        
        # self.cond_embedding = nn.Embedding(label_embed_classes, label_embed_dim) if label_embed_classes>0 != None else None
        self.cond_embedding = ls.Embedding( label_embed_classes, label_embed_dim, 
                                    embeddings_initializer=ks.initializers.RandomNormal( mean=0.0, stddev=1.0) ) if label_embed_classes>0 != None else None

        assert H == d_model

    def call(self, noise, conditional, cond_mask, diffusion_steps):
        
        #audio_cond: same shape as audio, audio_mask: same shape as audio but binary to be imputed where zero

        #noise #(b, seq, d)
        conditional = conditional * cond_mask       
        conditional_info = tf.concat([conditional, cond_mask], axis=1)  
          
        # diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_steps))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))
            
        x = noise        
        x = self.init_conv(x)   
        
        # Down blocks
        outputs = []
        outputs.append(x)
        for idx, layer in enumerate(self.d_layers):
            
            if isinstance(layer, ResidualBlock):
                x = layer((x,conditional_info,diffusion_step_embed))
            else:
                x = layer(x)
            
            outputs.append(x)
            
        # Center block
        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer((x,conditional_info,diffusion_step_embed))
            else:
                x = layer(x)
        x = x + outputs.pop() # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer((x,conditional_info,diffusion_step_embed))
                    else:
                        x = layer(x)
                    x = x + outputs.pop() # skip connection
            else:
                
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer((x,conditional_info,diffusion_step_embed))
                    else:
                        x = layer(x)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block
        
        x = self.norm(x)

        x = self.final_conv(x) 
        return x 

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SaShiMi
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped//3:]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    x = x + outputs.pop()
            else:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state

    def setup_rnn(self, mode='dense'):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.

        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%. 
                `linear` should be faster theoretically but is slow in practice since it 
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the diagonalization is numerically unstable
                (although we haven't encountered this case in practice), while `dense` should always be stable.
        """
        assert mode in ['dense', 'diagonal', 'linear']
        # for module in self.modules():
        #     if hasattr(module, 'setup_step'): module.setup_step(mode)
        
        for module in self.layers:
            if hasattr(module, 'setup_step'): module.setup_step(mode)

    @staticmethod
    def parse_config(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)

        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--in_channels", default=12, type=int ) #default value of 12 for ptbxl_248
        parser.add_argument("--out_channels", default=12, type=int )
        
        parser.add_argument("--pool", type= lambda x: [x,x] , default=[2,2], help="Pass an integer, this integer will be used as the pool rate in both the h,w dimensions" )
        parser.add_argument("--expand", default=2, type=int )
        parser.add_argument("--ff", default=2,  type=int )
        parser.add_argument("--glu", type = lambda x: bool(int(x)) , default=True)
        parser.add_argument("--unet", type = lambda x: bool(int(x)) , default=True)


        parser.add_argument("--dropout", default=0.0,  type=float) #not used

        parser.add_argument("--n_layers", default=6, type=int) #

        
        parser.add_argument("--diffusion_step_embed_dim_in", default=128, type=int)
        parser.add_argument("--diffusion_step_embed_dim_mid", default=512, type=int)
        parser.add_argument("--diffusion_step_embed_dim_out", default=512, type=int)

        parser.add_argument("--label_embed_dim", default=128, type=int)
        parser.add_argument("--label_embed_classes", default=71, type=int) # default=71

        parser.add_argument("--bidirectional", type = lambda x: bool(int(x)) , default=True)

        parser.add_argument("--s4_lmax", default=100, type=int)
        parser.add_argument("--s4_d_state", default=64, type=int)
        parser.add_argument("--s4_dropout", default=0.0, type=float) #not used
        parser.add_argument("--s4_bidirectional", type = lambda x: bool(int(x)) , default=True)

        parser.add_argument("--conv_channels_first", type = lambda x: bool(int(x)) , default=True, help="Provide 0 False to ensure that convolution operations use channels last formulation. Provides compatibility for windows cpu")

        config_model = parser.parse_known_args()[0]
        
        return config_model

class FFBlock(ls.Layer):

    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.dropout = dropout
        
        # input_linear = LinearActivation(
        #     d_model, 
        #     d_model * expand,
        #     transposed=True,
        #     activation='gelu',
        #     activate=True,
        # )
        # dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # output_linear = LinearActivation(
        #     d_model * expand,
        #     d_model, 
        #     transposed=True,
        #     activation=None,
        #     activate=False,
        # )

        # self.ff = ks.Sequential(
        #     input_linear,
        #     dropout,
        #     output_linear,
        # )

    def build(self, input_shape):

        input_linear = LinearActivation(
            self.d_model, 
            self.d_model * self.expand,
            transposed=True,
            activation='gelu',
            activate=True,
        )
        
        # dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        dropout = ks.layers.SpatialDropout2D(self.dropout) if self.dropout > 0.0 else ls.Layer()

        output_linear = LinearActivation(
            self.d_model * self.expand,
            self.d_model, 
            transposed=True,
            activation=None,
            activate=False,
        )

        self.ff = ks.Sequential(
            [input_linear,
            dropout,
            output_linear]
        )

        self.built = True
    
    def call(self, x):
        return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        # return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        return tf.squeeze( self.ff( x[...,tf.newaxis] ), -1), state

class ResidualBlock(ls.Layer):

    def __init__(
        self, 
        d_model, 
        layer,
        dropout,
        diffusion_step_embed_dim_out,
        in_channels,
        label_embed_dim,
        stride,
        conv_channels_first
    ):
        
        """
        Residual S4 block.

        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
        """
        super().__init__()        

        self.layer = layer
        # l = len(d_model) if isinstance(d_model, Iterable ) else 1
        # axis= list( range(-1, -1-l, -1) )
        self.norm = ls.LayerNormalization(axis=1, epsilon=1e-5, center=True, scale=True)
        self.dropout = ls.SpatialDropout2D(dropout) if dropout>0.0 else ls.Layer()

        self.fc_t = ls.Dense(d_model,
                                kernel_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_out),
                                bias_initializer= pytorch_he_uniform_init(diffusion_step_embed_dim_out )
                    )
        self.cond_conv = Conv(2*in_channels, d_model, kernel_size=stride, stride=stride, conv_channels_first=conv_channels_first)

         
    def call(self, input_data):
        """
        Input x is shape (B, d, L)
        diffusion_step_embed (B, d1)
        """
        x, cond, diffusion_step_embed = input_data
        
        # add in diffusion step embedding
        # part_t = self.fc_t(diffusion_step_embed).unsqueeze(2)
        part_t = tf.expand_dims( self.fc_t(diffusion_step_embed), axis=2) #(B, d, 1)
        z = x + part_t #(B, d, L)
        
        # Prenorm
        # z = tf.transpose( self.norm( tf.transpose(z, (0,2,1))), (0,2,1)) # (B, d, L)
        
        z = self.norm(z)

        cond = self.cond_conv(cond) #(B, d, L)
        
        z,_ = self.layer(z) #(B, d, L)
              
        z = z + cond #(B, d, L)
            
        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x

    
    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x

        # Prenorm
        z = self.norm(z)

        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)

        # Residual connection
        x = z + x

        return x, state

class DownPool(ls.Layer):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_input * expand
        self.pool = pool

        # self.linear = LinearActivation(
        #     d_input * pool,
        #     self.d_output,
        #     transposed=True,
        #     weight_norm=True,
        # )
        
    def build(self, input_shape):
        self.linear = LinearActivation(
            self.d_input * self.pool,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )
        
        # self.linear.build(input_shape)

        super(DownPool, self).build(input_shape)
        self.built = True

    def call(self, x):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            # x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            x = rearrange(tf.stack(state, axis=-1), '... h s -> ... (h s)')
            # x = x.unsqueeze(-1)
            x = x[...,tf.newaxis]
            x = self.linear(x)
            # x = x.squeeze(-1)
            x = tf.squeeze(x,-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []

class UpPool(ls.Layer):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        # self.linear = LinearActivation(
        #     d_input,
        #     self.d_output * pool,
        #     transposed=True,
        #     weight_norm=True,
        # )

    def build(self, input_shape):
        self.linear = LinearActivation(
            self.d_input,
            self.d_output * self.pool,
            transposed=True,
            weight_norm=True,
        )
        self.built = True

    def call(self, x):
        x = self.linear(x)
        
        if(self.causal):
            # x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
            x = tf.pad(  x[..., :-1], tf.constant( [[0,0],[0,0],[1,0]] ) )
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            # x = x.unsqueeze(-1)
            x = x[..., tf.newaxis]
            x = self.linear(x)
            # x = x.squeeze(-1)
            x = tf.squeeze(x, -1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            # state = list(torch.unbind(x, dim=-1))
            state= list( tf.unstack(x, axis=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        # state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = tf.zeros( ( batch_shape + (self.d_output, self.pool)) ) # (batch, h, s)
        # state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        state = list(tf.unstack(state, axis=-1)) # List of (..., H)
        return state

