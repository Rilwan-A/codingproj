import tensorflow as tf
import random
import numpy as np
from tensorflow import keras as ks
ls = tf.keras.layers
import tensorflow_addons as tfa
import math
import copy
import einops
from tensorflow.python.ops import gen_array_ops
import logging

from typeguard import typechecked
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

        
# Utility funcs : Loss / Metrics
def metrics_get(metric_name):
    """
        Helper methods to get Loss/Metric classes for metrics used in SSSD-S4 paper
            mse, rmse, mae, cprs, mre

    Returns:
        _type_: Union[Loss, Metric]_description_
    """
    if metric_name == 'rmse':
        return ks.metrics.get('RootMeanSquaredError')
    
    elif metric_name == 'crps':
        return ContinuousRankedProbabilityScore()
    
    elif metric_name == 'mre':
        return MeanRelativeError()

    elif metric_name == 'mae':
        return ks.metrics.get('MeanAbsoluteError')

    elif metric_name == 'mse':
        return ks.metrics.get('MeanSquaredError')
    
    elif metric_name == 'logreturn':
        return LogReturnLoss()
    
    else:
        metric =  ks.metrics.get(metric_name)
        assert isinstance(metric, ks.metrics.Metric), 'metric named passed must reflect a valid metric class and not a function'
        return metric

class ContinuousRankedProbabilityScore(tf.keras.metrics.Metric):
    def __init__(self, name='crps', **kwargs):
        super(ContinuousRankedProbabilityScore, self).__init__(name=name, **kwargs)
        
        self.li_target = []
        self.li_forecast = []
        self.li_loss_mask = []

        self.CRPS = self.add_weight(name='crps', initializer='zeros')

        self.mean_scaler = kwargs.get('mean_scaler', 0.0)
        self.scaler = kwargs.get('scaler', 1.0)

        self.update_state_non_masked_input = True
        
    def update_state(self, target, forecast, loss_mask, **kwargs): #, eval_points, mean_scaler, scaler):

        self.li_target.append( target )
        self.li_forecast.append( forecast )
        self.li_loss_mask.append( loss_mask )

    def result(self):
        
        if len(self.li_target) == 0:
            return 0.0

        target = tf.concat(self.li_target,0)
        forecast = tf.concat(self.li_forecast,0)
        loss_mask = tf.concat(self.li_loss_mask,0)

        target = target * self.scaler + self.mean_scaler
        forecast = forecast * self.scaler + self.mean_scaler

        quantiles = np.arange(0.05, 1.0, 0.05)
        # denom = self.calc_denominator(target, self.eval_points)
        denom = tf.reduce_sum( tf.math.abs( tf.broadcast_to( target[...,tf.newaxis], forecast.shape ) * loss_mask ) )
        
        for i in range(len(quantiles)):
            q_pred = []
            for j in range(len(forecast)):
                q_pred.append(
                    tf.constant(
                        np.quantile(forecast[j: j + 1], quantiles[i], axis=-1),
                         self.compute_dtype)) # axis 1 implies we are getting ith quantile of channel info -> seems slightly wrong, shouldn't it be ith quantile within a channel over a time chunk
            q_pred = tf.concat(q_pred, 0)
            q_loss = self.quantile_loss(target, q_pred, quantiles[i])
            self.CRPS.assign_add( q_loss / denom )
        
        return self.CRPS / len(quantiles)
    
    def reset_state(self):
        self.CRPS = self.add_weight(name='crps', initializer='zeros')    
        self.li_target = []
        self.li_forecast = []
        
    def quantile_loss(self, target, forecast, q: float) -> float:
        # return 2 * tf.reduce_sum(tf.math.abs((forecast - target) * ((target <= forecast) * 1.0 - q)))
        
        # forecast = forecast[:, :, tf.newaxis] #tf.broadcast_to( forecast, target.shape )

        a = (forecast - target)
        b = tf.cast(target <= forecast, a.dtype) * 1.0
        c = (b - q)

        return 2 * tf.reduce_sum(tf.math.abs(a * c))

class MeanRelativeError(tf.keras.metrics.Metric):
    def __init__(self, name='mre', **kwargs):
        super(MeanRelativeError, self).__init__(name=name, **kwargs)
        self.reset_state()

    def update_state(self, y_true, y_pred):

        relative_errors = tf.math.divide_no_nan(
            tf.abs(y_true - y_pred), tf.abs(y_true)
        )
        relative_errors = tf.reshape(relative_errors, (-1) )

        # self.li_mre.append(relative_errors)
        
        batch_sum = tf.reduce_sum(relative_errors)
        
        self.sum += tf.cast(batch_sum, tf.float64)
        self.count += tf.size(relative_errors, out_type=tf.float64)

    def result(self):
        
        if self.count>0:
            # mre = tf.concat(self.li_mre, 0)
            # # mre_avg = tf.reduce_mean( mre )
            
            # batch_sum = tf.reduce_sum(mre)
            # self.sum += tf.cast(batch_sum, tf.float64)
            # self.count += tf.sze(mre)
            
            mre_avg = self.sum / self.count
        else:
            mre_avg = 0.0

        return mre_avg
    
    def reset_state(self):
        self.sum = tf.constant(0.0, dtype=tf.float64)
        self.count =  tf.constant(0.0, dtype=tf.float64)
        return None

# class LogReturnLoss():
# class LogReturnLoss():
#     def __init__(self, name='logreturn', **kwargs):
#         # super(LogReturnLoss, self).__init__(name=name, **kwargs)
#         self.reset_state()
#         self.update_state_non_masked_input = True
#         self.name = name
class LogReturnLoss(tf.keras.metrics.Metric):
    def __init__(self, name='logreturn', **kwargs):
        super(LogReturnLoss, self).__init__(name=name, **kwargs)
        self.reset_state()
        self.update_state_non_masked_input = True

    def update_state(self, logreturn_daily, pred_logreturn, loss_mask, **kwargs):

        """_summary_

        Args:
            logreturn (tf.Tensor): Tensor of shape (b, c, seq_len)
            target_return (tf.Tensor): Tensor of shape  (b, c, 1) 
            pred_logreturn (tf.Tensor):  Tensor of shape (b, c, seq_len, samples)
            loss_mask (tf.Tensor):  Tensor of shape (b, c, seq_len, samples)
 

        Returns:
            _type_: _description_
        """        
        # Aim is to compute MSE loss of predicted return and true return over holiday period
        # perform a reduce_sum on filtered pred_logreturn to get log return over holida
        # Then exponentiate it to get return
        # Then MSE with target return
        
        # Sum along the sequence dimension to get total log return for specific holiday seq for each stock in exchange
        target_return = kwargs['target']
        scaler = kwargs.get('scaler')
        
        loss_mask = tf.cast(loss_mask, tf.bool)
        
        B, C, L, S = pred_logreturn.shape
        pred_logreturn = einops.rearrange(pred_logreturn, 'b c l s -> (b l s) c', b=B, l=L, s=S, c=C)
        pred_logreturn = scaler.inverse_transform( pred_logreturn - 0.05 ) 
        pred_logreturn = einops.rearrange(pred_logreturn, '(b l s) c-> b c l s', b=B, l=L, s=S, c=C)
        
        pred_logreturn1 = tf.where( loss_mask, pred_logreturn, tf.zeros_like(pred_logreturn))
        
        pred_logreturn2 = tf.reduce_sum( pred_logreturn1, axis= -2 ) #( b, c, samples)
        
        # Filtering for relevant channels
        loss_mask_channel = tf.reduce_any(loss_mask, axis=-2) 
        
        pred_logreturn3 = tf.boolean_mask(pred_logreturn2, loss_mask_channel )
        pred_return = tf.math.exp(pred_logreturn3)
        
        # target_return = tf.broadcast_to(target_return, (B, C, S) )   
        target_return = tf.tile(target_return, (1, 1, S) )        
        target_return1 = tf.boolean_mask(target_return, loss_mask_channel)
        
        # Handling errors relating to misidentified holidays or stocks in the same stock index having different holidays
        mask_is_nan = tf.math.is_nan(target_return1)
        target_return1 = tf.boolean_mask(target_return1, ~mask_is_nan)
        pred_return = tf.boolean_mask(pred_return, ~mask_is_nan)
        
        loss = tf.keras.metrics.mean_squared_error(target_return1, pred_return)
        
        
        self.sum += tf.cast(loss, tf.float64)
        self.count += tf.convert_to_tensor(1.0, self.count.dtype)

    def result(self):
        
        if self.count>0:
            # mre = tf.concat(self.li_mre, 0)
            # # mre_avg = tf.reduce_mean( mre )
            
            # batch_sum = tf.reduce_sum(mre)
            # self.sum += tf.cast(batch_sum, tf.float64)
            # self.count += tf.sze(mre)
            
            result = self.sum / self.count
        else:
            result = 0.0

        return result
    
    def reset_state(self):
        self.sum = tf.constant(0.0, dtype=tf.float64)
        self.count =  tf.constant(0.0, dtype=tf.float64)
        return None
