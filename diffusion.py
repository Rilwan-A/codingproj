import tensorflow as tf
from tensorflow import keras as ks
ls = ks.layers


import numpy as np
import argparse 
import math
import einops
from util_layers import std_normal

import random
#region Masking
class MaskingMixinAlvarez():

    def get_cond_mask(self, shape=None, mask_method=None, missing_k=None, observed_mask=None, cond_mask=None):

        #NOTE: the mask_method here is the same for each element in a batch should be improved such that each datum has different mask

        if mask_method == 'mnr':
            cond_mask = self.get_cond_mask_mnr(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )

        elif mask_method == 'bm':
            cond_mask = self.get_cond_mask_bm(shape[1:], missing_k)
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )

        elif mask_method == 'rm':
            cond_mask = self.get_cond_mask_rm(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )
              
        elif mask_method == 'tf':
            cond_mask = self.get_cond_mask_tf(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )

        elif mask_method == 'bm_channelgroup':
            """bm style masking but only applied to specific groups of channels
                Under yahoo stocks;
                    use to mask out values for a specific exchange to replicate missing data for a holiday                    
            """
            cond_mask = get_cond_mask_bm_channel(shape, missing_k)
            
        cond_mask = tf.cast( cond_mask, self.compute_dtype)
        
        # observed_mask logic should be moved into function and applied beforehand instead of after
        observed_mask = tf.cast( observed_mask, self.compute_dtype) if (observed_mask is not None) else tf.ones_like(cond_mask) 
                
        cond_mask = cond_mask * observed_mask

        return cond_mask

    #TODO: Extend this function to work with an observed_mask e.g. not assuming all data is already visisble
    #TODO: simply change mask to observed mask, and allow it to be passed as an argument, if not passed then assume its an array of all ones
    # - if observed then use the CSDI method using select_top_k, if not then use the original method
    
    def get_cond_mask_rm(self, shape, k):
        """Get mask of random points (missing at random) across channels based on k,
        where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
        for channel in range(shape[0]):
            perm = np.random.permutation( shape[1] )
            idx = perm[0:k]
            mask[channel, :][idx] = 0

        return tf.constant( mask, self.compute_dtype)

    def get_cond_mask_mnr(self, shape, k):
        """Get mask of random segments (non-missing at random) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
        length_index = np.arange(shape[1])
        # list_of_segments_index = np.split(length_index, k)
        list_of_segments_index = np.array_split(length_index, k)
        for channel in range(shape[0]):
            s_nan = random.choice(list_of_segments_index)
            mask[channel, :][s_nan[0]:s_nan[-1] + 1] = 0

        return tf.constant( mask, self.compute_dtype)

    def get_cond_mask_bm(self, shape, k):
        """Get mask of same segments (black-out missing) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
        length_index = np.arange(shape[1])
        # list_of_segments_index = np.split(length_index, k)
        list_of_segments_index = np.array_split(length_index, k)
        s_nan = random.choice(list_of_segments_index)
        for channel in range(shape[0]):
            mask[channel, :][s_nan[0]:s_nan[-1] + 1] = 0

        return tf.constant( mask, self.compute_dtype)
 
    def get_cond_mask_tf(self, shape, k):
        """Get mask of same segments (black-out missing) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
                
        s_nan = np.arange( shape[1]-k, shape[1] )

        for channel in range(shape[0]):
            mask[channel, :][s_nan[0]:s_nan[-1] + 1] = 0
        
        return tf.constant( mask, self.compute_dtype)

    def get_loss_mask(self, cond_mask,
            tgt_mask_method:str, observed_mask=None,
            eval_all_timesteps=False ):

        # If not observed mask supplied assumed all series was observed
        observed_mask = observed_mask if observed_mask else tf.ones_like(cond_mask)

        target_mask = tf.ones_like(cond_mask)

        # Masking 
        if tgt_mask_method == 'tgt_all_remaining':
            # This covers any region which was not a part of the conditional region and was observed
            target_mask = target_mask * (1-cond_mask)
            target_mask = target_mask * observed_mask
        elif tgt_mask_method in ['tgt_random_from_remaining','mix']:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        # The tgt_mask is all observed values that are not part of the conditional mask
        if eval_all_timesteps:
            target_mask = einops.repeat(target_mask, 'b ... -> (b L) ...', L=self.T)
        
        return tf.cast(target_mask,tf.bool)

class MaskingMixinCSDI():

    def get_cond_mask(self, shape, mask_method, missing_k, observed_mask=None, cond_mask=None ):

        # If not observed mask supplied assumed all series was observed
        observed_mask = observed_mask if (observed_mask is not None) else tf.ones( shape )

        # Masking methods from Alvarez
        if mask_method == 'mnr':
            cond_mask = self.get_cond_mask_mnr(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )

        elif mask_method == 'bm':
            mask = self.get_cond_mask_bm(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )

        elif mask_method == 'rm':
            cond_mask = self.get_cond_mask_rm(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )
            
        elif mask_method == 'tf':
            cond_mask = self.get_cond_mask_tf(shape[1:], missing_k)
            # mask = tf.transpose( mask, perm=(1, 0) )
            cond_mask = tf.tile( cond_mask[tf.newaxis,...], (shape[0], 1, 1) )

        elif mask_method == 'bm_channelgroup':
            """bm style masking but only applied to specific groups of channels
                Under yahoo stocks;
                    use to mask out values for a specific exchange to replicate missing data for a holiday                    
            """
            cond_mask = get_cond_mask_bm_channel(shape, missing_k)
            
        cond_mask = tf.cast( cond_mask, self.compute_dtype)
        
        # observed_mask logic should be moved into function and applied beforehand instead of after
        observed_mask = tf.cast( observed_mask, self.compute_dtype) if (observed_mask is not None) else tf.ones_like(cond_mask) 
                
        cond_mask = cond_mask * observed_mask

        return cond_mask

    def get_cond_mask_rm(self, shape, k):
        """Get mask of random points (missing at random) across channels based on k,
        where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
        # length_index = np.arange(shape[0])  # lenght of series indexes
        for channel in range(shape[0]):
            # perm = torch.randperm(len(length_index))
            # perm = tf.random.shuffle( tf.range(len(length_index)) )
            perm = np.random.permutation( shape[1] )
            idx = perm[0:k]
            mask[channel, :][idx] = 0

        return tf.constant( mask, self.compute_dtype)

    def get_cond_mask_mnr(self, shape, k):
        """Get mask of random segments (non-missing at random) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
        length_index = np.arange(shape[1])
        # list_of_segments_index = np.split(length_index, k)
        list_of_segments_index = np.array_split(length_index, k)
        for channel in range(shape[0]):
            s_nan = random.choice(list_of_segments_index)
            mask[channel, :][s_nan[0]:s_nan[-1] + 1] = 0

        return tf.constant( mask, self.compute_dtype)

    def get_cond_mask_bm(self, shape, k):
        """Get mask of same segments (black-out missing) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
        length_index = np.arange(shape[1])
        # list_of_segments_index = np.split(length_index, k)
        list_of_segments_index = np.array_split(length_index, k)
        s_nan = random.choice(list_of_segments_index)
        for channel in range(shape[0]):
            mask[channel, :][s_nan[0]:s_nan[-1] + 1] = 0

        return tf.constant( mask, self.compute_dtype)

    def get_cond_mask_tf(self, shape, k):
        """Get mask of same segments (black-out missing) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
        as per ts imputers"""
        k = int(k)
        mask = np.ones(shape)
                
        s_nan = np.arange( shape[1]-k, shape[1] )

        for channel in range(shape[0]):
            mask[channel, :][s_nan[0]:s_nan[-1] + 1] = 0
        
        return tf.constant( mask, self.compute_dtype)

    def get_loss_mask(self, cond_mask, tgt_mask_method, observed_mask=None, eval_all_timesteps=False, **kwargs):

        # If not observed mask supplied assumed all series was observed
        observed_mask = observed_mask if observed_mask else tf.ones_like(cond_mask)

        target_mask = tf.ones_like(cond_mask)
        
        if tgt_mask_method == 'tgt_all_remaining':
            # This mask has True for any region which 
            # was not a part of the conditional region and which was observed
            target_mask = target_mask * (1-cond_mask)
            target_mask = target_mask * observed_mask
        
        elif tgt_mask_method in ['tgt_random_from_remaining','mix']:
            raise NotImplementedError
        
        else:
            raise NotImplementedError
        
        if eval_all_timesteps:
            target_mask = einops.repeat(target_mask, 'b ... -> (b L) ...', L=self.num_steps)

        return tf.cast(target_mask,tf.bool)
#endregion

def get_cond_mask_bm_channel(shape, k="0 29 29 103 103 124" ):
        """Get mask of same segments (black-out missing) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
        as per ts imputers
        
        In this version k should be a list of tuples of indexes indiciating the 
            start and end  channel that each segment mask will be apply.
            Here masking is applied to one of a preset group of channels
        
        k = '0 11 12 39 45 50' Defines 3 groups of channels to choose from for masking
            channel1:[0-11], channel2: [12-39], channel3: [40-50] indicating each idx belongs to channel1       
        """

        batch, c, seq_len = shape
        mask = np.ones(shape)
        # length_index = np.arange(seq_len)
               
        # gathering the stop and start indexes for the channel groups that will be masked
        li_k = list(map(int, k.split(' ')))
        li_channelgroup_idxs = [ list(range(sidx, eidx)) for sidx, eidx in zip(li_k[::2],li_k[1::2]) ]
        
        # Sample channels to mask corresponding to each stock index
        li_exchange_to_mask = random.choices(li_channelgroup_idxs, k=batch)
        
        # Sample lengths of for segments to be masked - sample between 1 and 9 to reflect holidaay lengths
            # NOTE: minimum sample lenth is 2, since we are evaluating returns so a 1 day holiay is 2 days no return
        # li_seg_len = [ int(random.triangular(2, 8+1)) for idx in range(batch) ]
        li_seg_len = random.choices( [2,3,4], weights=[0.81, 0.143, 0.04 ] , k=batch )
        
        # starting index for each segment to be masked
        # NOTE: the max position of seq_len-l-1 ensures that the lastelement in sequence is not a holiday
            # This allows us to calculate return over a holiday from prices
        li_seg_sidx = [  random.choice(range(0, seq_len-l-1)) for l in li_seg_len]
        
        for batch_idx in range(batch):
            s_xchng = li_exchange_to_mask[batch_idx][0]
            e_xchng = li_exchange_to_mask[batch_idx][1]
            
            s_seg = li_seg_sidx[batch_idx]
            e_seg = s_seg + li_seg_len[batch_idx]            
            
            mask[batch_idx,  s_xchng:e_xchng+1, s_seg:e_seg] = 0

        # mask = np.transpose(mask, (0, 2, 1)) # (batch, channels, seq_len)        
        
        return mask
        
#region Diffussion
class DiffusionAlvarez(ls.Layer, MaskingMixinAlvarez):

    def __init__(self, T, beta_0, beta_T, 
                mask_method="bm", tgt_mask_method='tgt_all_remaining' , missing_k=50  ):
        
        super(DiffusionAlvarez, self).__init__()
        
        self.T = T
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.Beta, self.Alpha, self.Alpha_bar, self.Sigma = self.calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)
                
        self.mask_method = mask_method 
        self.tgt_mask_method = tgt_mask_method
        self.missing_k = missing_k

    def calc_diffusion_hyperparams(self, T, beta_0, beta_T):
        """
        Compute diffusion process hyperparameters

        Parameters:
        T (int):                    number of diffusion steps
        beta_0 and beta_T (float):  beta schedule start/end value, 
                                    where any beta_t in the middle is linearly interpolated
        
        Returns:
        a dictionary of diffusion hyperparameters including:
            T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
            These cpu tensors are changed to cuda tensors on each individual gpu
        """

        Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
            # / (1-\bar{\alpha}_t)
        Sigma = np.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
        
        return tf.constant(Beta), tf.constant(Alpha), tf.constant(Alpha_bar), tf.constant(Sigma)
        
    def call(self, model, audio, cond_mask, eval_all_timesteps=False):
        """
        Compute the training loss of epsilon and epsilon_theta

        Parameters:
        model (torch network):            the wavenet model
        X (torch.tensor):               training data, shape=(batchsize, 1, length of audio) 
        
        Returns:
        training loss
        """
 
        T, Alpha_bar = self.T, self.Alpha_bar

        B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
        B = B if B else 1
        
        if eval_all_timesteps == True:
            # TODO: make sure CSDI models use this loss func
            # NOTE: using all timesteps is implemented in CSDI but not in Alvarez paper
            # diffusion_steps = tf.range(0, T)  # randomly sample diffusion steps from 1~T            
            # diffusion_steps = einops.repeat(diffusion_steps, 'l -> B 1 l', b=B, c=C )
        
            # diffusion_steps = tf.expand_dims( diffusion_steps[:, tf.newaxis, tf.newaxis], axis=(B, 1, 1) )
            
            li_epsilon_theta = []
            li_epsilon = []

            for idx in range(self.T):
                # diffusion_steps = tf.constant( [idx]*B, dtype=tf.int64 )

                # diffusion_steps = tf.experimental.numpy.random.randint(0, T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
                diffusion_steps = tf.cast( tf.fill( (B, 1, 1), idx), dtype=tf.int64 )

                # epsilon = std_normal(audio.shape)
                epsilon = std_normal((B,C,L))
                
                if self.tgt_mask_method == "tgt_all_remaining":
                    epsilon = audio * cond_mask + epsilon * (1 - cond_mask)
                
                transformed_X = tf.cast( tf.math.sqrt( tf.gather(Alpha_bar, diffusion_steps) ), audio.dtype) * audio + tf.cast(tf.math.sqrt(
                            1 - tf.gather(Alpha_bar, diffusion_steps)), audio.dtype) * epsilon  # compute x_t from q(x_t|x_0)

                cond = audio*cond_mask 
                epsilon_theta = model(transformed_X, cond, cond_mask, tf.squeeze(diffusion_steps,-1))  # predict \epsilon according to \epsilon_\theta

                li_epsilon_theta.append(epsilon_theta)
                li_epsilon.append(epsilon)
            
            epsilon_theta = tf.concat(li_epsilon_theta, axis=0)
            epsilon = tf.concat(li_epsilon, axis=0)

        else:
            #Training Loop
            diffusion_steps = tf.experimental.numpy.random.randint(0, T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T

            # epsilon = std_normal(audio.shape)
            epsilon = std_normal((B,C,L), audio.dtype )
            
            if self.tgt_mask_method == "tgt_all_remaining":
                epsilon = audio * cond_mask + epsilon * (1 - cond_mask)
            
            transformed_X = (
                tf.cast( tf.math.sqrt( tf.gather(Alpha_bar, diffusion_steps) ), audio.dtype) * audio +
                tf.cast(tf.math.sqrt(1 - tf.gather(Alpha_bar, diffusion_steps)), audio.dtype) * epsilon)  # compute x_t from q(x_t|x_0)

            cond = audio*audio
            epsilon_theta = model(
                transformed_X, cond, cond_mask,
                tf.squeeze(diffusion_steps,-1) )  # predict \epsilon according to \epsilon_\theta

        return epsilon_theta, epsilon

    def sampling(self, model, size, observed, cond_mask, pred_samples=1, dtype=tf.float32):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        model (torch network):            the wavenet model
        size (tuple):                   size of tensor to be generated, 
                                        usually is (number of audios to generate, channels=1, length of audio)
        cond_mask (b, d, seq)
        
        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        T, Alpha, Alpha_bar, Sigma = self.T, self.Alpha, self.Alpha_bar, self.Sigma
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(size) == 3
        dtype = dtype if dtype is not None else self.compute_dtype
        
        print('begin sampling, total number of reverse steps = %s' % T)
 
        # x = std_normal(size)
        x = std_normal( [ size[0]*pred_samples, *size[1:] ], dtype=dtype ) #b, d, seq_len
        observed = tf.cast(tf.repeat(observed, pred_samples, axis=0), dtype)
        cond_mask = tf.cast(tf.repeat(cond_mask, pred_samples, axis=0), dtype)
        
        for t in range(T - 1, -1, -1):
            
            if self.tgt_mask_method == 'tgt_all_remaining':
                x = x * (1 - cond_mask) + observed * cond_mask
            diffusion_steps = (t * tf.ones((size[0]*pred_samples, 1), dtype=tf.dtypes.float32))  # use the corresponding reverse step
            epsilon_theta = model(x, observed, cond_mask, diffusion_steps)  # predict \epsilon according to \epsilon_\theta
            
            # update x_{t-1} to \mu_\theta(x_t)
            A_t = tf.cast(Alpha[t], x.dtype)
            Ab_t = tf.cast(Alpha_bar[t], x.dtype)
            epsilon_theta = tf.cast(epsilon_theta,x.dtype)
            x = (x - (1 - A_t) / tf.math.sqrt(1 -Ab_t) * epsilon_theta) / tf.math.sqrt(A_t)
            if t > 0:
                x = x + tf.cast(Sigma[t],x.dtype)  * std_normal( [ size[0]*pred_samples, *size[1:] ] ) # add the variance term to x_{t-1}
        
        # Converting back to (b, samples, d, seq)
        x = einops.rearrange(x, '(b s) ... -> b ... s', s=pred_samples )

        return x

    @staticmethod
    def parse_config(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)

        #masking        
        parser.add_argument("--mask_method", default="rm", choices=["rm","mnr","bm","tf","bm_channelgroup","custom"])
        parser.add_argument("--tgt_mask_method", default="tgt_all_remaining", choices=[ "tgt_all_remaining", "tgt_random_from_remaining" ,"mix"], help="")
        parser.add_argument("--missing_k", default='90', type=str,
            help="For rm mask method: The number of elements in the sequence to mask for each channel \n\
                    For mnr mask method: The number of segments to split the sequence into, from which one segment will become the target. This target segment may differ for each channel\n\
                    For bm mask method: The number of segments to split the sequence into, from which one segment will become the target. This target segment is the same for each channel\n\
                    For tf mask method: The number of elements in the sequence to forecast" )
        parser.add_argument("--T", default=200, type=int )
        parser.add_argument("--beta_0", default=0.0001, type=float )
        parser.add_argument("--beta_T", default=0.02, type=float )

        config_diffusion = parser.parse_known_args()[0]

        return config_diffusion
        
class DiffusionCSDI(ls.Layer, MaskingMixinCSDI):

    def __init__(self, 
                num_steps,
                beta_start,
                beta_end,
                mask_method,
                tgt_mask_method,
                schedule,
                missing_k):
        
        super(DiffusionCSDI, self).__init__()
        
        self.num_steps = num_steps 
        self.beta_start = beta_start 
        self.beta_end = beta_end 
        self.schedule = schedule
        self.mask_method  = mask_method
        self.tgt_mask_method = tgt_mask_method
        self.missing_k = missing_k

        # Creating Diffusion Params
        self.beta, self.alpha_hat, self.alpha_tf = self.calc_diffusion_hyperparams(
            self.num_steps, self.beta_start, self.beta_end, self.schedule
        )
    
    def calc_diffusion_hyperparams(self, num_steps, beta_start, beta_end, schedule):
        """
        Compute diffusion process hyperparameters

        Parameters:
        T (int):                    number of diffusion steps
        beta_0 and beta_T (float):  beta schedule start/end value, 
                                    where any beta_t in the middle is linearly interpolated
        
        Returns:
        a dictionary of diffusion hyperparameters including:
            T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
            These cpu tensors are changed to cuda tensors on each individual gpu
        """

        if schedule == "quad":
            beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        
        elif schedule == "linear":
            beta = np.linspace(beta_start, beta_end, num_steps)

        alpha_hat = 1 - beta
        alpha = np.cumprod(alpha_hat)
        alpha_tf = tf.constant(alpha)[:,tf.newaxis, tf.newaxis]

        # alpha_tf = tf.cast( tf.constant(alpha), tf.float64)
        return tf.constant(beta, dtype=tf.float32), tf.constant(alpha_hat, tf.float32), tf.cast(alpha_tf, tf.float32)
          
    def call(self, model, observed_data, cond_mask, eval_all_timesteps=False):

        B, K, L = observed_data.shape

        # observed_timesteps = tf.range(0, L, size=(B,L))
        # observed_timesteps = tf.tile( tf.range(0,L)[tf.newaxis], (B, 1))

        if eval_all_timesteps == True:
            # NOTE: using all timesteps is implemented in CSDI but not in Alvarez paper
            # diffusion_steps = tf.range(0, self.num_steps, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
            # diffusion_steps = tf.range(0, self.num_steps)  # randomly sample diffusion steps from 1~T
            # diffusion_steps = einops.repeat(diffusion_steps, 'b -> (repeat b )', repeat=B ) # B*L
            # observed_data = einops.repeat( observed_data, ' b ... -> (b repeat) ...', repeat=L )
            # cond_mask = einops.repeat( cond_mask, ' b ... -> (b repeat) ...', repeat=L )

            li_noise_theta = []
            li_noisy_target = []
            for idx in range(self.num_steps):
                diffusion_steps = tf.constant( [idx]*B, dtype=tf.int64 )

                # current_alpha = self.alpha_tf[t]  # (B,1,1)
                current_alpha = tf.gather( self.alpha_tf, diffusion_steps )

                # noise = torch.randn_like(observed_data).to(self.device)
                noise = std_normal( observed_data.shape )
                noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
                cond_obs = (cond_mask * observed_data)[:, tf.newaxis]

                noisy_target = ((1 - cond_mask) * noisy_data)       # (B, K, L)
                inp = tf.concat([cond_obs, noisy_target[:, tf.newaxis]], axis=1)  # (B,2,K,L)
                noise_theta = model(inp, cond_mask, diffusion_steps)  # (B,K,L)  ###### AND HERE!!!!

                li_noise_theta.append(noise_theta)
                li_noisy_target.append(noisy_target)
            
            noise_theta = tf.concat(li_noise_theta, axis=0)
            noisy_target = tf.concat(li_noisy_target, axis=0)

            # TODO: fix the eval_all_timesteps logic
        else:
            #Training Loop
            diffusion_steps = tf.experimental.numpy.random.randint(0, self.num_steps, size=(B,))  # randomly sample diffusion steps from 1~T

            current_alpha = tf.cast( tf.gather( self.alpha_tf, diffusion_steps ), self.compute_dtype) # (B,1,1)
            noise = std_normal( observed_data.shape, self.compute_dtype )
            
            noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
            cond_obs = (cond_mask * observed_data)[:, tf.newaxis]

            noisy_target = ((1 - cond_mask) * noisy_data)
            inp = tf.concat([cond_obs, noisy_target[:, tf.newaxis]], axis=1)  # (B,2,K,L)

            noise_theta = model(inp, cond_mask, diffusion_steps)  # (B,K,L)  ###### AND HERE!!!!

        return noise_theta, tf.cast( noisy_target, noise_theta.dtype )
        

    def sampling(self, model, size, observed, cond_mask, pred_samples=1, dtype=tf.float32):
        
        dtype = dtype if dtype is not None else self.compute_dtype

        B, K, L = size
                           

        # extending in batch dimension for sampling in one pass through
        current_sample = std_normal( (pred_samples*B, K, L), dtype)
        observed = tf.cast(tf.repeat(observed, pred_samples, axis=0), dtype)
        cond_mask = tf.cast(tf.repeat(cond_mask, pred_samples, axis=0), dtype)

        for t in range(self.num_steps - 1, -1, -1):
            cond_obs = (cond_mask * observed)[:, tf.newaxis]
            
            # if self.tgt_mask_method == 'tgt_all_remaining':
            noisy_target = ((1 - cond_mask) * current_sample )[:, tf.newaxis]

            inp = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
            
            predicted = model(inp, cond_mask, tf.constant([t], dtype=tf.int64))

            alpha_hat_t = tf.cast(self.alpha_hat[t], dtype)
            alpha_t = tf.cast(self.alpha_tf[t], dtype)

            coeff1 = 1 / alpha_hat_t ** 0.5
            coeff2 = (1 - alpha_hat_t) / (1 - alpha_t) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)
            
            if t > 0:
                # noise = torch.randn_like(current_sample)
                noise = std_normal((B*pred_samples,K,L) , predicted.dtype)
                
                alpha_tm1 = tf.cast(self.alpha_tf[t-1], dtype)
                betat = tf.cast(self.beta[t], dtype)
                sigma = (
                                (1.0 - alpha_tm1) / (1.0 - alpha_t) * betat
                        ) ** 0.5
                current_sample += noise * sigma
        
                
        current_sample = einops.rearrange(current_sample, '(b s) ... -> b ... s', s=pred_samples )

        return current_sample

    @staticmethod
    def parse_config(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)

        #masking     
        parser.add_argument("--mask_method", default="rm", 
            choices=[ "random", "mix","rm","mnr","bm","tf", "bm_channelgroup","custom"])   
        
        parser.add_argument("--tgt_mask_method", default="tgt_all_remaining",
                choices=[ "tgt_all_remaining", "tgt_random_from_remaining" ,"mix"], help="")

        parser.add_argument("--missing_k", default='90', type=str,
            help="For rm mask method: The number of elements in the sequence to mask across all channels \n\
                    For mnr mask method: The number of segments to split the sequence into, from which one segment will become the target. This target segment may differ for each channel\n\
                    For bm mask method: The number of segments to split the sequence into, from which one segment will become the target. This target segment is the same for each channel\n\
                    For tf mask method: The number of elements in the sequence to forecast" )
        
        parser.add_argument("--schedule", default="quad", choices=["quad","linear"])

        parser.add_argument("--num_steps", type=int, default=50 )
        parser.add_argument("--beta_start", type=float, default=0.0001 )
        parser.add_argument("--beta_end", type=float ,default=0.5 )
              
        config_diffusion = parser.parse_known_args()[0]
        return config_diffusion
        
#endregion

