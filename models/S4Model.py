import numpy as np

from tqdm import tqdm
import pickle

import math

from functools import partial
from scipy import special as ss

from einops import rearrange, repeat
import opt_einsum as oe
import copy

contract = oe.contract
contract_expression = oe.contract_expression

import tensorflow as tf
from tensorflow import keras as ks
ls = ks.layers
import tensorflow_addons as tfa

from collections.abc import Iterable   # import directly from collections for Python < 3.3
from util_layers import WeightNormalization, pytorch_he_normal_init

# _c2r = torch.view_as_real
# _r2c = torch.view_as_complex
# _c2r = lambda x: tf.stack( [ tf.math.real(x), tf.math.imag(x)], axis=-1 )
# _r2c = lambda x: tf.dtypes.complex(x[...,0], x[...,1] )

# _conj = lambda x: tf.concat([x, tf.math.conj(x)], axis=-1)
# _resolve_conj = lambda x: tf.math.conj(x)

def _c2r(x): 
    return tf.stack( [ tf.math.real(x), tf.math.imag(x)], axis=-1 )

def _r2c(x):
    return tf.complex(x[...,0], x[...,1] )

def _conj(x):
    return tf.concat([x, tf.math.conj(x)], axis=-1)

def _resolve_conj(x):
    return tf.math.conj(x)

''' Standalone CSDI + S4 imputer for random missing, non-random missing and black-out missing.
The notebook contains CSDI and S4 functions and utilities. However the imputer is located in the last Class of
the notebook, please see more documentation of use there. Additional at this file can be added for CUDA multiplication 
the cauchy kernel.'''

log = tf.get_logger()

@tf.function(reduce_retracing=False,
             jit_compile=True)
def cauchy_slow(v, z, w):
    """
    v, w: (..., N)
    z: (..., L)
    returns: (..., L)
    """
    cauchy_matrix = tf.expand_dims(v,-1) / (tf.expand_dims(z,-2) - tf.expand_dims(w,-1)) # (... N L)
    outp = tf.reduce_sum( cauchy_matrix, axis=-2)
    
    return outp

""" S4 Classes """
class S4Layer(ls.Layer):
    #S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(self, features, lmax, d_state=64, dropout=0.0,
            bidirectional=True, layer_norm=True, transposed=True):
        super().__init__()
        self.s4  = S4(d_model=features, 
                            d_state=d_state, 
                            l_max=lmax, 
                            bidirectional=bidirectional,
                            transposed=transposed)
        
        l = len(features) if isinstance(features, Iterable ) else 1
        axis= list( range(-1, -1-l, -1) )
        
        self.norm_layer = ls.LayerNormalization(axis=axis, epsilon=1e-5, center=True,
                            scale=True) if layer_norm else ls.Layer() #ls.Identity()
        self.dropout = ls.SpatialDropout1D(dropout) if dropout>0 else ls.Layer() #ls.Identity()

    def call(self, x):
        # x [b, d, seq]
        
        # x = tf.transpose(x, perm=(1, 2, 0) ) #batch, d, seq
        
        xout, _ = self.s4(x)                  #batch, d, seq
        
        # Dropout requires channels last
        xout = tf.transpose(xout, perm=(0, 2, 1))   # (b, seq, d)
        xout = self.dropout(xout)                   # (b, seq, d)
        xout = tf.transpose(xout, perm=(0, 2, 1))   # (b, d, seq)

        xout = xout + x # skip connection           # (b, d, seq)
        
        xout = tf.transpose(xout, perm=(0,2,1))     # (b, seq, d)
        xout = self.norm_layer(xout)                # (b, seq, d)
        xout = tf.transpose(xout, perm=(0,2,1))     # (b, d, seq)
        
        return xout

class S4(ls.Layer):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            hyper_act=None, # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True, #(b d seq) if self.transposed else (b seq d)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
        ):


        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: is the input of shape (b d seq) if transposed==True else (b seq d)

        Other options are all experimental and should not need to be configured
        """
        super(S4, self).__init__()

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.kernel_args = kernel_args
        self.l_max = l_max
        self.initializer = initializer
        self.weight_norm = weight_norm
        self.postact = postact
        self.hyper_act = hyper_act
        self.hyper = hyper_act is not None

        # Pointwise
        self.activation = Activation(activation, -2)
        
        if dropout == 0.0:
            self.dropout = ls.Layer()
        else:
            self.dropout =  ks.Sequential( [ ls.Permute( (2,1) ), ls.SpatialDropout1D(dropout) , ls.Permute( (2,1) )] ) 
    
    def build(self, input_shape):  # Create the state of the layer (weights)

        channels = copy.deepcopy(self.channels)

        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(self.hyper_act)

        self.D = tf.Variable( tf.random.normal((channels, self.h)) , name = 'D', trainable=False )

        if self.bidirectional:
            channels *= 2
        self.kernel = HippoSSKernel(self.h, N=self.n, L=self.l_max, channels=channels, **self.kernel_args ) 
                
        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            initializer=self.initializer,
            activation=self.postact,
            activate=True,
            weight_norm=self.weight_norm) #self.Variable X 2
        super(S4, self).build(input_shape)

    def call(self, u): # absorbs return_output and transformer src mask
        """
        u: (b d seq) 
        state: (d n) never needed unless you know what you're doing

        Returns: same shape as u

        """     
        L = u.shape[-1]
        H = u.shape[-2]
        l = tf.convert_to_tensor([L], tf.int32)
        # Compute SS Kernel
        k = self.kernel(l=l) #  (B H L)

        # Convolution
        if self.bidirectional:
            _k = rearrange(k, '(s c) h l -> s c h l', s=2, h=H, l=L)
            k0, k1 = _k[0], _k[1]
            
            # k = F.pad(k0, (0, L)) \
            #         + F.pad(k1.flip(-1), (L, 0)) \
            
            k =  tf.pad( k0, tf.constant([[0,0],[0,0], [0,L]]) )\
                + tf.pad( tf.reverse(k1,[-1]), tf.constant([[0,0],[0,0], [L,0]]) )
        
        
        C = 1

        ft_len = tf.constant([2*L])
        k_f = tf.signal.rfft(tf.cast(k,'float32'), ft_len ) # (C H L)
        
        u = tf.cast(u,'float32')        
        u_f = tf.signal.rfft( u, ft_len)  # (B H L)
        y_f = tf.einsum('bhl,chl->bchl', u_f, k_f)
        y = tf.signal.irfft(y_f, ft_len)[..., :L] # (B C H L)
        
        # Compute D term in state space equation - essentially a skip connection
        y = y + tf.einsum('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2, l=L, h=H)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l', h=H, c=C , l=L) #(b d l)
        y = self.activation(y)
        y = self.dropout( y )

        y = self.output_linear( y )
          
        return y, None
        
    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + tf.expand_dims(u,-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            
            #original
            y = tf.squeeze( self.output_linear(tf.expand_dims(y,-1)), -1)

        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)

""" HiPPO Classes """
class HippoSSKernel(ls.Layer):
  
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=1,
        measure="legs",
        rank=1,
        channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        trainable=None, # Dictionary of options to train various HiPPO parameters
        lr=None, # Hook to set LR of hippo parameters differently
        length_correction=True, # Multiply by I-A|^L after initialization; can be turned off for initialization speed
        hurwitz=False,
        tie_state=False, # Tie parameters of HiPPO ODE across the H features
        precision=1, # 1 (single) or 2 (double) for the kernel
        resample=False,  # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = tf.double if self.precision == 2 else tf.float32
        cdtype = tf.complex64 if dtype == tf.float32 else tf.complex128
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = tf.random.uniform( [self.H], dtype=dtype ) * (
            tf.math.log(dt_max) - tf.math.log(dt_min)
        ) + tf.math.log(dt_min)

        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        C_i, C_j = tf.random.normal( (self.channels, self.H, self.N // 2), stddev=math.sqrt(1/2),  dtype=dtype ), tf.random.normal( (self.channels, self.H, self.N // 2), stddev=math.sqrt(1/2) , dtype=dtype )
        C = tf.complex(C_i, C_j)
        
        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction
        )

    def call(self, l=None):
        rate = tf.reshape( tf.convert_to_tensor(self.rate), [1] )
        k = self.kernel(rate=rate, l=l)
        return k

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)

class SSKernelNPLR(ls.Layer):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """
    def __init__(
        self,
        L, w, P, B, C, log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
        verbose=False,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*

        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)

        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.length_correction = length_correction

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        # assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        assert w.shape[-1] == P.shape[-1] == B.shape[-1] == C.shape[-1]
        # self.H = log_dt.size(-1)
        # self.N = w.size(-1)
        self.H = log_dt.shape[-1]
        self.N = w.shape[-1]

        # Broadcast everything to correct shapes
        # C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)
        self._C = tf.broadcast_to(C, tf.broadcast_dynamic_shape( C.get_shape(), (1, self.H, self.N))) # (H, C, N)

        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        self.L = tf.convert_to_tensor(self.L)

        self.C = _c2r(_resolve_conj(C))

        self.train = False
        if trainable is None: 
            self.dict_trainable = {}
        elif trainable == False: 
            self.dict_trainable = {}
        elif trainable == True:
            self.dict_trainable, self.train = {}, True
        else:
            self.dict_trainable = trainable
        

        self.log_dt = log_dt 
        self.B = _c2r(B)
        self.P = _c2r(P)

        if self.hurwitz:
            log_w_real = tf.math.log(-tf.math.real(w) + 1e-3) # Some of the HiPPO methods have real part 0
            w_imag = tf.math.image(w)

            self.Q = None

            self.log_w_real = log_w_real
            self.w_imag = w_imag

        else:
            #TODO {rilwan-adewoyin} - Figure out what the lr and 0.0 arguments do in the original self.register calls below

            self.w = _c2r(w)
            self.Q = _c2r( _resolve_conj(tf.identity(P)) )

    def build(self, input_shape):
        
        super().build(input_shape)

        self.C = tf.Variable(self.C, name='C', trainable=True)

        self.log_dt = tf.Variable(self.log_dt, trainable=self.dict_trainable.get('dt', self.train), name='log_dt' )
        self.B = tf.Variable(self.B, trainable=self.dict_trainable.get('B', self.train),name='B') #Variable:0
        self.P = tf.Variable(self.P, trainable=self.dict_trainable.get('P', self.train),name='P') #Variable:0

        # Cache Fourier nodes every time we set up a desired length
        # self.L = L
        if self.L is not None:
            _l = tf.convert_to_tensor( self.L, dtype=tf.int32 )
            self.omega, self.z = self._omega(_l, dtype=self._C.dtype, cache=True)
            
        if self.hurwitz:
            self.log_w_real = tf.Variable(self.log_w_real, trainable=self.dict_trainable.get('A', False), name='log_w_real') #Variable:0
            self.w_imag = tf.Variable(self.w_imag, trainable=self.dict_trainable.get('A', self.train),name='w_imag') #Variable:0

        else:
            self.w = tf.Variable(self.w, trainable=self.dict_trainable.get('A', self.train), name='w') #Variable
            self.Q = tf.Variable(self.Q, trainable=self.dict_trainable.get('P', self.train), name='Q') #Variable:0

        if self.length_correction:
            self._setup_C()

    def call(self, rate=1.0, l=None):
        state=None
        
        # calc L, rate 
        rate, L, L_float = self.call1(rate, l)
        
        # conditional loop
        self.omega, self.z, self.L = self.call2(rate, L_float)
        
        # create loads of vars        
        dt, B, C, P, Q, w, omega, z = self.call3(rate)
                
        # create loads of vars        
        dt, v, w = self.call4(state, w, Q, P, B,  C, dt)
        
        # cauchy
        r = cauchy_slow(v, z, w)
                                
        # Low-rank Woodbury correction
        k_B, k_state = self.call5(r, omega, state, L, dt  )

        return k_B

    @tf.function(
        jit_compile=True,
        reduce_retracing=False,
    )
    def call1(self, rate, l):

        state=None
        
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        
        L = l
        
        # assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = tf.cast( self.L / rate, tf.int32)

        # Increase the internal length if needed
        L_float = tf.cast(L, rate.dtype)
        
        return rate, L, L_float

    @tf.function(
        jit_compile=False,
        reduce_retracing=True
    )
    def call2(self, rate, l_float):
        L_float = l_float
        self.L = tf.cast(self.L, L_float.dtype )
        min_l = tf.reshape(rate * L_float, ())
        iterations = tf.cast( tf.experimental.numpy.log2( min_l/self.L ) +1, tf.int32 )
        iterations = tf.math.maximum( iterations, tf.zeros_like(iterations) )

        C = tf.identity( self.C)
       
        self.L = tf.reshape(self.L, [1] )
        for idx in tf.range(iterations):
            tf.autograph.experimental.set_loop_options(    
                shape_invariants=[
                    ( self.L ,tf.TensorShape([1]) ),
                    ( self.omega, tf.TensorShape([None, 2])),
                    ( self.z, tf.TensorShape([None, 2])),
                    ( C, tf.TensorShape([None, None, None, 2]))
                ])
                  
            self.L = tf.reshape(self.L, [])
            self.L, self.omega, self.z, C = self.double_length_graphmode(self.L, self.omega, self.z, C )
            self.L = tf.reshape(self.L, [1,])
            
        self.L = tf.cast(self.L, tf.int32 )
        self.L = tf.reshape(self.L, [] )
        self.C.assign(C)
        
        return self.omega, self.z, self.L
        
    
    @tf.function(
        reduce_retracing=False,
        jit_compile=True
        )
    def call3(self, rate):
        dt = tf.cast( tf.math.exp(self.log_dt), rate.dtype ) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = tf.math.conj(P) if self.Q is None else _r2c(self.Q)
        w = self._w()
        

        rate = tf.reshape(rate,())
        if rate==1.0:
            # Use cached FFT nodes
            omega = _r2c(self.omega)
            z =_r2c(self.z)  # (..., L)     
            
        else:
            _lr = self.L//tf.cast(rate,self.L.dtype)
            omega, z = self._omega(_lr, dtype=B.dtype, cache=False)

        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)
            
        return dt, B, C, P, Q, w, omega, z

    @tf.function(
        reduce_retracing=False,
        jit_compile=True)
    def call4(self, state, w, q, p, b, c, dt ):

        Q, P, B, C = q, p, b, c
        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            # s = _conj(state) if state.size(-1) == self.N else state # (B H N)
            s = _conj(state) if state.shape[-1] == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / tf.expand_dims(dt, -1) + sA / 2
            s = s[..., :self.N]

            B = tf.concat([s, B], axis=-3)  # (s+1, H, N)

        # Incorporate dt into A
        dt = tf.cast(dt, tf.complex64)
        # w = w * tf.expand_dims( dt, -1)  # (H N)
        w = tf.einsum('ij,i->ij', w, dt) # (H N)
        
        # Stack B and p, C and q for convenient batching
        B = tf.concat((B, P), axis=-3) # (s+1+r, H, N)
        C = tf.concat((C, Q), axis=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = tf.einsum('bij,cij->bcij',B,C) # (s+1+r, c+r, H, N)
        
        return dt, v, w

    @tf.function(
        reduce_retracing=False,
        jit_compile=True)
    def call5(self, r, omega, state, l ,dt):
        L =l
        
        r = tf.einsum('bscl,c->bscl',r,dt)
        
        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
            
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            
            r11 = tf.linalg.inv(tf.eye(self.rank) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            
            k_f = r00 - tf.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = tf.signal.irfft(k_f)  # (S+1, C, H, L)
        # Truncate to target length
        k = k[..., :tf.reshape(L, ())]

        
        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)
        return k_B, k_state

    # @torch.no_grad()
    def double_length(self):
        # if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)

    def double_length_graphmode(self, self_l,  omega, z, C):
        # if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self.L, self.omega, self.z, C = self._setup_C_graphmode(C, double_length=True)
        return self.L, self.omega, self.z, C
            
    # @torch.no_grad()
    def _setup_C(self, double_length=False):
        """ Construct C~ from C

        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        
        self.step_params, self.dA, self.dB = self._setup_state()
        dA_L = power(self.L, self.dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h n m, c h n -> c h m", dA_L, C_)
        
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        C_ = _c2r(C_)
        C_ = tf.stop_gradient(C_)
        self.C.assign(C_)

        if double_length:
            self.L *= 2
            
            self.omega, self.z = self._omega(tf.convert_to_tensor(self.L,tf.int32), dtype=C.dtype, cache=True)
    
    # @torch.no_grad()
    def _setup_C_graphmode(self, C, double_length=False):
        """ Construct C~ from C

        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(C)
        
        # self.step_params, self.dA, self.dB = self._setup_state())
        self.step_params, self.dA, self.dB = self._setup_state_graphmode(C)
        
        dA_L = power_graphmode(self.L, self.dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = tf.einsum("hnm,chn->chm",dA_L,C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        C_ = _c2r(C_)
        C_ = tf.stop_gradient(C_)
        # self.C.copy_(C_)
        self.C.assign(C_)

        if double_length:
            self.L *= 2
            self.omega, self.z = self._omega( tf.cast(self.L, tf.int32), dtype=C.dtype, cache=True)
            return self.L, self.omega, self.z, self.C
        
        return self.L, self.C

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self.step_params = self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = tf.expand_dims( tf.eye( 2*self.N, dtype=C.dtype ),-2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        # u = C.new_ones(self.H)
        u = tf.ones(self.H, dtype=C.dtype)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)
        
        return self.step_params, self.dA, self.dB

    def _setup_state_graphmode(self, C):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self.step_params = self._setup_linear()
        
        state = tf.expand_dims( tf.eye( 2*self.N, dtype=C.dtype ),-2) # (N 1 N)
        
        dA = self._step_state_linear_graphmode(C, state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        
        u = tf.ones(self.H, dtype=C.dtype)
        dB = self._step_state_linear_graphmode(C, u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)
        
        return self.step_params, self.dA, self.dB
    
    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()

        B = tf.cast( _r2c(self.B), w.dtype)  # (H N)
        P = tf.cast( _r2c(self.P), w.dtype)
        Q = tf.math.conj(self.P) if self.Q is None else tf.cast(_r2c(self.Q), w.dtype)
        

        # Prepare Linear stepping
        dt = tf.cast( tf.math.exp(self.log_dt), w.dtype )
        
        D = 2.0 / tf.expand_dims(dt, -1) - w
        D = D**-1 

        R = (tf.eye(self.rank, dtype=w.dtype) +
                2* tf.cast( tf.math.real( contract('r h n, h n, s h n -> h r s', Q, D, P) ), w.dtype)
                 ) # (H r r)

        Q_D = rearrange(Q*D, 'r h n -> h r n')
                
        R = tf.linalg.solve(R, Q_D) # (H r N)
        
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": 2.0 / tf.expand_dims(dt, axis=-1) + w, # (H N)
        }
        return self.step_params
        
    def _omega(self, L, dtype=tf.complex64, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        
        omega = tf.exp( tf.constant(-2j * np.pi, dtype=dtype) / tf.cast(L, dtype=dtype) )  # \omega_{2L}
        
        omega = omega ** tf.cast( tf.range(0, tf.cast(L,tf.int32) // 2 + 1), dtype=dtype)
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            self.omega =  _c2r(omega)
            self.z = _c2r(z)
            return self.omega, self.z
        return omega, z

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -tf.math.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros( (self.H, self.N), dtype=C.dtype)

        # step_params = self.step_params.copy()
        # step_params = tf.identity(self.step_params)
        step_params = {k:tf.identity(v) for k,v in self.step_params.items()}

        if state.get_shape()[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.get_shape()[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)
       

        _ = [*state.get_shape()]
        _[-2] = Q.get_shape()[1]
        tf.broadcast_to(state, _)
        new_state = E * state - contract_fn(P, Q, tf.broadcast_to(state, _) )# (B H N)
        

        new_state = new_state + 2.0 * B * tf.expand_dims(u,-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _step_state_linear_graphmode(self, C,u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        # C = _r2c(C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros( (self.H, self.N), dtype=C.dtype)

        # step_params = self.step_params.copy()
        # step_params = tf.identity(self.step_params)
        step_params = {k:tf.identity(v) for k,v in self.step_params.items()}

        if state.get_shape()[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.get_shape()[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)
        
        _ = [*state.get_shape()]
        _[-2] = Q.get_shape()[1]
        tf.broadcast_to(state, _)
        new_state = E * state - contract_fn(P, Q, tf.broadcast_to(state, _) )# (B H N)
        

        new_state = new_state + 2.0 * B * tf.expand_dims(u,-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state

    def setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self.step_params, self.dA, self.dB = self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        # I = torch.eye(self.dA.size(-1)).to(dA_L)
        I = tf.cast( tf.eye( self.dA.get_shape()[-1] ), dA_L.dtype)
        C = _conj(_r2c(self.C)) # (H C N)

        dC = tf.squeeze(tf.math.linalg.solve(
                I - tf.transpose(dA_L,(-1, -2)),
                C[...,tf.newaxis], #.expand_dims(C,-1),
                ), -1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = tf.linalg.eig(self.dA)
            V_inv = tf.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", ks.metrics.mean_squared_error(V @ tf.linalg.diag(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        # N = C.size(-1)
        # H = C.size(-2)
        N = C.shape[-1]
        H = C.shape[-2]

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode !='linear':
            N *= 2

            if self._step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N), # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N), # self.dC.shape
            batch_shape + (H, N),
        )

        # state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        state = tf.zeros( (*batch_shape, H, N), dtype=C.dtype)
        return state

    def step(self, u, state):
        """ Must have called self.setup_step() and created state with self.default_state() before calling this """

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

""" simple nn.Module components """
def Activation(activation=None, axis=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return ls.Layer() #ls.Identity()
    elif activation == 'tanh':
        return ls.Activation('tanh')
    elif activation == 'relu':
        return ls.Activation('relu')
    elif activation == 'gelu':
        return ls.Activation('gelu')
    elif activation in ['swish', 'silu']:
        return ls.Activation('swish')
    elif activation == 'glu':
        return GLU(axis=axis)
    elif activation == 'sigmoid':
        return ls.Activation('sigmoid')
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

class GLU(ls.Layer):
    r"""
        glu(input, dim=-1) -> Tensor

        The gated linear unit. Computes:

        .. math ::
            \text{GLU}(a, b) = a \otimes \sigma(b)

        where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
        is the sigmoid function and :math:`\otimes` is the element-wise product between matrices.

        See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

        Args:
            input (Tensor): input tensor
            dim (int): dimension on which to split the input. Default: -1
    """
    def __init__(self, axis:int=-1):
        super(GLU, self).__init__()
        self.axis= axis
        self.sigmoid = ls.Activation( tf.nn.sigmoid )
    
    @tf.function(
        reduce_retracing=True,
        jit_compile=True 
        )
    def call(self, x):
        # assert x.shape[self.axis] % 2 == 0, "axis size must be divisible by 2"
        x1, x2 = tf.split(x, 2, axis=self.axis)
        outp = x1 * self.sigmoid(x2)
        return outp

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")
    
    if name == 'uniform':
        assert nonlinearity == 'relu', "Default keras implementation is only identical to pytorch implementation if linearity is ReLU "
        initializer = ks.initializers.HeUniform()
    elif name == 'normal':
        assert nonlinearity == 'relu', "Default keras implementation is only identical to pytorch implementation if linearity is ReLU"
        initializer = ks.initializers.HeNormal()
    elif name == 'xavier':
        initializer = ks.initializers.GlorotNormal()
    elif name == 'zero':
        initializer = ks.initializers.Zeros()
    elif name == 'one':
        initializer = ks.initializers.Ones()
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")
    
    return initializer

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    if activation == 'glu': 
        d_output *= 2
                                    
    linear = ls.Dense(d_output, use_bias=bias, bias_initializer='zeros' if zero_bias_init else None)

    # Weight norm
    if weight_norm:
        linear = WeightNormalization(linear )

    if transposed:
        linear = ks.Sequential([ls.Permute((2,1)), linear, ls.Permute((2,1)) ])
    
    if activate and activation is not None:
        activation = Activation(activation, axis=-2 if transposed==True else -1)
        linear = ks.Sequential([linear, activation])

    return linear


""" Misc functional utilities """
def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    # x = b.unsqueeze(-1) # (..., N, 1)
    x = tf.expand_dims(b,-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        # AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        AL = tf.eye(A.shape[-1], dtype=A.dtype)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        _x = A_ @ _x
        # x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        x = tf.concat([x, _x], dim=-1)
        if not done: A_ = A_ @ A_

    assert x.get_shape()[-1] == L

    if c is not None:
        # x = torch.einsum('...nl, ...n -> ...l', x, c)
        x = tf.einsum('...nl, ...n -> ...l', x, c)

    # x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = tf.cast( tf.eye(A.shape[-1]), A.dtype ) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.get_shape()[-1] - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.get_shape()[-1] > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, tf.squeeze(v,-1)


def power_graphmode(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = tf.cast( tf.expand_dims( tf.eye(A.shape[-1]),0), A.dtype ) # , dtype=A.dtype, device=A.device)

    # powers = [A]
    l = 1
    idx = tf.constant(0)
    iterations = tf.cast(tf.experimental.numpy.log2(L), tf.int32)
    powers = tf.TensorArray(A.dtype, iterations+1, A.dtype, clear_after_read=False )
    powers = powers.write(0,A)
    L = tf.reshape(L, [1,])
    for idx in tf.range(iterations+1):
        tf.autograph.experimental.set_loop_options(
            shape_invariants= [
                ( l ,tf.TensorShape([]) ),
                ( L,tf.TensorShape([1,]) ),
                ( I ,tf.TensorShape([None, None, None]) ),   
                # ( powers, tf.TensorArraySpec(element_shape=tf.TensorShape([None, None, None]) ) )  
            ]
            
            )
        
        if tf.equal( tf.math.floormod(L, 2), 1):
            I = powers.read(idx) @ I
        
        L //= 2
        
        if tf.not_equal(L, 0.0):
            l *= 2

            pow = powers.read(idx)
            powers = powers.write( idx+1, pow @ pow )
            

    if v is None: 
        powers.close()
        # powers.mark_used()
        return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.get_shape()[-1] - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.get_shape()[-1] > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, tf.squeeze(v,-1)

""" HiPPO utilities """
def nplr(measure, N, rank=1, dtype=tf.float32):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == tf.float32 or tf.complex64
    if measure == 'random':
        dtype = tf.complex64 if dtype == tf.float32 else tf.complex128
        w = -tf.math.exp(tf.random.normal((N//2))) + 1j*tf.random.normal((N//2))
        P = tf.random.normal( (rank, N//2), dtype=dtype)
        B = tf.random.normal( (N//2), dtype=dtype)
        V = tf.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    #TODO: check A and B do not have requires_grad == False
    A = tf.convert_to_tensor(A, dtype=dtype) # (N, N)
    B = tf.convert_to_tensor(B, dtype=dtype)[:,0] # (N,)

    
    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + tf.math.reduce_sum(tf.expand_dims(P,-2)*tf.expand_dims(P,-1), axis=-3)
    
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    w, V = tf.linalg.eig(AP) # (..., N) (..., N, N)

    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2]
    V = V[..., 0::2]

    # V_inv = V.conj().transpose(-1, -2)
    V_inv = tf.transpose( tf.math.conj(V), [1, 0], conjugate=True )
    
    # B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    # P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P
    
    B = contract('ij, j -> i', V_inv, tf.cast(B, V.dtype))
    P = contract('ij, ...j -> ...i', V_inv, tf.cast(P,V.dtype)) # V^* P
    
    return w, P, B, V

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

def rank_correction(measure, N, rank=1, dtype=tf.float32):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        # P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
        P = tf.math.sqrt(.5 + tf.range(N, dtype=dtype)) # (1, N)
        P = tf.expand_dims(P,0)
    elif measure == 'legt':
        assert rank >= 2
        # P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P = tf.math.sqrt(1+2*tf.range(N, dtype=dtype)) # (N)
        # P0 = P.clone()
        P0 = tf.identity(P)
        P0[0::2] = 0.
        # P1 = P.clone()
        P1 = tf.identity(P)
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * tf.ones( (1, N), dtype=dtype)
        P = .5**.5 * tf.ones((1, N), dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype) # (N)
        # P0 = P.clone()
        P0 = tf.identity(P)
        P0[0::2] = 0.
        # P1 = P.clone()
        P1 = tf.identity(P)
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    else: raise NotImplementedError

    d = P.shape[0]
    if rank > d:
        P = tf.concat([P, tf.zeros( (rank-d, N), dtype=dtype)], axis=0) # (rank N)
    return P

def bilinear(dt, A, B=None):
    """
    dt: (...) timescales
    A: (... N N)
    B: (... N)
    """
    N = A.shape[-1]
    # I = torch.eye(N).to(A)
    I = tf.cast( tf.eye(N), A.dtype)
    # A_backwards = I - dt[:, None, None] / 2 * A
    # A_forwards = I + dt[:, None, None] / 2 * A
    
    A_backwards = I - dt[:, tf.newaxis, tf.newaxis] / 2 * A
    A_forwards = I + dt[:, tf.newaxis, tf.newaxis] / 2 * A
    
    if B is None:
        dB = None
    else:
        dB = dt[..., tf.newaxis] * tf.squeeze( tf.linalg.solve(
            A_backwards, B[...,tf.newaxis])
        ,-1) # (... N)

    dA = tf.linalg.solve(A_backwards, A_forwards)  # (... N N)
    return dA, dB
        
# def get_torch_trans(heads=8, layers=1, channels=64):
#     encoder_layer = nn.TransformerEncoderLayer(
#         d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
#     return nn.TransformerEncoder(encoder_layer, num_layers=layers)

