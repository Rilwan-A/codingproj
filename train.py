import os
os.environ['HDF5_USE_FILE_LOCKING'] ='FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import warnings
warnings.filterwarnings('ignore')

import argparse
from argparse import ArgumentParser
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import pickle
from tensorflow import keras as ks
import yaml
import glob
import copy
import shutil
import tensorflow_addons as tfa
from datasets.datasets import Dset_V2
from util_losses import metrics_get
from contextlib import nullcontext
from __init__ import MAP_NAME_DIFFUSION, MAP_NAME_DSET, MAP_MNAME_MODEL
import math
from transformers.optimization_tf import WarmUp
import einops
from sklearn.preprocessing import QuantileTransformer

# Utility funcs: paramter conversion
def config_trainer_param_adjust(config_trainer, batched_dset_len=None ):
    """Determines the number of epochs to train for depending on the size of the dset
        and 'config_trainer.steps_per_epoch'
        
    Args:
        config_trainer (_type_): _description_
        batched_dset_len (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    epoch_set_by_user = (config_trainer.epochs is not None)
    
    # converting num_iters to epochs        
    if not epoch_set_by_user:
        assert batched_dset_len is not None
        assert hasattr(config_trainer,'n_iters')
        config_trainer.epochs = math.ceil(config_trainer.n_iters / batched_dset_len)
    
    if config_trainer.steps_per_epoch is None:
        config_trainer.steps_per_epoch =  int( batched_dset_len)
        if not epoch_set_by_user: 
            config_trainer.epochs = int( config_trainer.epochs *  (batched_dset_len/config_trainer.steps_per_epoch) )           
    elif not (config_trainer.steps_per_epoch).is_integer():
        # If steps_per_epoch is a float, representing a proportion of dataset size, then we scale up epochs 
        config_trainer.steps_per_epoch =  int( batched_dset_len*config_trainer.steps_per_epoch)
        if not epoch_set_by_user: 
            config_trainer.epochs = int( config_trainer.epochs *  (batched_dset_len/config_trainer.steps_per_epoch) ) 
    else:
        # Ensuring steps_per_epoch is an int
        config_trainer.steps_per_epoch = int(config_trainer.steps_per_epoch)
        if not epoch_set_by_user:
            config_trainer.epochs = int( config_trainer.epochs *  (batched_dset_len/config_trainer.steps_per_epoch) ) 

    return config_trainer

class ImputationTrainerCallback(ks.callbacks.Callback):

    def __init__(self, dir_ckpt, exp_name, **kwargs):

        super().__init__()
        self.dir_ckpt = dir_ckpt
        self.exp_name = exp_name
        
    def on_predict_end(self, logs=None):
        """Called at the end of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        
        dict_metrics = { name: m.result().numpy().item()
                            for name,m in 
                            self.model.map_mname_predmetric.items()
                            }
        
        dir_outp = os.path.join(os.path.join(self.dir_ckpt, self.exp_name, 'test' ))
        
        with open( os.path.join(dir_outp,'pred_metrics.yaml') , 'w' ) as f:
            yaml.safe_dump( dict_metrics, f)
            
class ImputationTrainer(tf.keras.Model):

    def __init__(self, model, diffusion, base_loss:str=None, train_metrics:str=None, 
                val_metrics:str=None, pred_metrics:str=None, 
                pred_samples:int=1,
                eval_all_timestep=False,
                grad_accum=None,
                **kwargs):
        
        super(ImputationTrainer, self ).__init__()

        self.model = model
        self.diffusion = diffusion
        self.eval_all_timestep = eval_all_timestep
        self.grad_accum = grad_accum
        self.scaler = kwargs.get('scaler', None)
                
        # Loss trackers
        self.base_loss = ks.losses.get(base_loss) if base_loss else tf.keras.losses.mean_squared_error
        self.train_loss_tracker = ks.metrics.Mean(name="loss")
        self.val_loss_tracker = ks.metrics.Mean(name="loss")
        
        # Setting up metrics
        self.train_metric_names = sorted(train_metrics.split('_')) if train_metrics else []
        self.val_metric_names = sorted(val_metrics.split('_')) if val_metrics else []
        self.pred_metric_names = sorted(pred_metrics.split('_')) if pred_metrics else []
        self.pred_samples = pred_samples

        # TODO: reimplement out way to initalize this to only create based on which of train, val, test is being run
        self.map_mname_trainmetric = {
            'train_'+metric_name:metrics_get(metric_name)
            for metric_name in self.train_metric_names
        }

        self.map_mname_valmetric = {
            metric_name:metrics_get(metric_name)
            for metric_name in self.val_metric_names
        }

        self.map_mname_predmetric = {
            metric_name:metrics_get(metric_name)
            for metric_name in self.pred_metric_names
        }

    def call(self,  audio=None, cond_mask=None, eval_all_timesteps=False ):
        if cond_mask is not None:
            cond_mask = tf.cast(cond_mask, dtype=audio.dtype)
        
        epsilon_theta, epsilon = self.diffusion( self.model, audio, cond_mask, eval_all_timesteps )
        
        return epsilon_theta, epsilon
    
    def unpack_batch(self, batch):
        
        if isinstance(batch, tf.Tensor):
            audio = batch
            observed_mask = tf.ones_like(audio)
            cond_mask = None
            loss_mask = None
            target = None
            other = {}
            
        
        elif type(batch) == dict:
            audio = batch['data']
            observed_mask = batch.pop('observed_mask', tf.ones_like(audio))
            cond_mask = batch.pop('cond_mask', None)
            loss_mask = batch.pop('loss_mask', None)
            # target_return = batch.pop('target_return', None)
            # target_price = batch.pop('target_price', None)
            other = batch
                        
        return audio, observed_mask, cond_mask, loss_mask, other
    
    def step(self, batch, eval_all_timesteps=False, step_name='val'):
        
        audio, observed_mask, cond_mask, loss_mask, other = self.unpack_batch(batch)
        
        audio = (audio * tf.cast(observed_mask,audio.dtype))
          
        cond_mask = cond_mask if (cond_mask is not None) else self.diffusion.get_cond_mask(audio.shape, 
                        self.diffusion.mask_method,
                        self.diffusion.missing_k,
                        observed_mask) #This is same as cond_mask in CSDI
        
        
        loss_mask = loss_mask if (loss_mask is not None) else self.diffusion.get_loss_mask( cond_mask = cond_mask,
                                    tgt_mask_method=self.diffusion.tgt_mask_method,
                                    observed_mask=None,
                                    eval_all_timesteps=eval_all_timesteps)

        # data pass through
        with tf.GradientTape() if (step_name == 'train') else nullcontext() as tape:
            
            # Forward Step
            epsilon_theta, epsilon = self(audio, cond_mask, eval_all_timesteps=eval_all_timesteps)
            
            # Loss
            loss = self.base_loss( tf.boolean_mask( epsilon_theta, tf.cast(loss_mask,tf.bool)),
                                  tf.boolean_mask(epsilon, tf.cast(loss_mask,tf.bool)) )

            if self.mixed_precision is True:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        # Gradient update - conditional on training
        if (step_name == 'train'):

            # Mixed Precision Update
            if self.mixed_precision is True:
                scaled_gradients =  tape.gradient(scaled_loss, self.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:            
                # Compute gradients
                gradients = tape.gradient(loss, self.trainable_variables)   
            
            if self.grad_accum is None: 
                # No Gradient Accumulation
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables), skip_gradients_aggregation=True)
            else:
                # Initiate accumulated gradients at 0
                if self._train_counter == 0:
                    self.accumulated_gradients = [ tf.zeros_like(var) for var in self.trainable_variables]

                # Gradient Accumulation logic
                if self._train_counter % self.grad_accum or self._train_counter==0:
                    # Aggregate gradients
                    self.accumulated_gradients = [
                        (acc_grad + grad) if grad is not None else None for acc_grad, grad in zip( self.accumulated_gradients, gradients)
                    ]
                                    
                else:
                    # Apply gradients
                    self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.trainable_variables), skip_gradients_aggregation=True)
                    self.accumulated_gradients = [ tf.zeros_like(var ) for var in self.trainable_variables]
            
        return loss, epsilon_theta, epsilon, loss_mask

    def train_step(self, batch):
        
        loss, epsilon_theta, epsilon, loss_mask = self.step(batch, step_name='train')
        
        # Update metrics and loss
        self.train_loss_tracker.update_state(loss)
        
        epsilon_masked = tf.boolean_mask(epsilon, loss_mask)
        epsilon_theta_masked = tf.boolean_mask(epsilon_theta, loss_mask)
        for k in self.map_mname_trainmetric:
            self.map_mname_trainmetric[k].update_state(epsilon_masked, epsilon_theta_masked)
        
        dict_loss_and_metrics = { 
            **{ self.train_loss_tracker.name:loss },
             **{ name: m.result() for name,m in self.map_mname_trainmetric.items() } }
        
        return dict_loss_and_metrics
             
    def test_step(self, batch):
        
        loss, epsilon_theta, epsilon, loss_mask = self.step(batch, eval_all_timesteps=self.eval_all_timestep, step_name='val'  )
        
        self.val_loss_tracker.update_state(loss)
        epsilon_masked = tf.boolean_mask(epsilon, loss_mask)
        epsilon_theta_masked = tf.boolean_mask(epsilon_theta, loss_mask)
        for k in self.map_mname_valmetric:
            self.map_mname_valmetric[k].update_state(epsilon_masked, epsilon_theta_masked)

        dict_loss_and_metrics = { **{ self.val_loss_tracker.name:self.val_loss_tracker.result() }, **{ name: m.result() for name,m in self.map_mname_valmetric.items() } }
        
        return dict_loss_and_metrics
        
    def predict_step(self, batch):
        
        if type(batch) == dict:
            w_idx = batch.pop('w_idx', None)
            s_idx = batch.pop('s_idx', None)
            len_ = batch.pop('len', None)
            exchange_index = batch.pop('exchange_index', None)
        
        audio, observed_mask, cond_mask, loss_mask, other = self.unpack_batch(batch)
                         
        cond_mask = cond_mask if (cond_mask is not None) else self.diffusion.get_cond_mask(audio.shape, 
                                        self.diffusion.mask_method, 
                                        self.diffusion.missing_k,
                                        observed_mask=observed_mask)

        loss_mask = loss_mask if (loss_mask is not None) else self.diffusion.get_loss_mask(
                                        cond_mask = cond_mask,
                                        tgt_mask_method=self.diffusion.tgt_mask_method,
                                        observed_mask=observed_mask)
        
        # Generating prediction
        generated_audio = self.diffusion.sampling(
            self.model,
            audio.shape,
            observed=(audio * tf.cast(observed_mask,audio.dtype)),
            cond_mask=cond_mask,
            pred_samples=self.pred_samples
        ) # (bs, d, seq, pred_sample )
        
        # Creating expanded versions to match m samples per prediction
        audio_expanded = tf.broadcast_to( audio[..., tf.newaxis], generated_audio.shape ) 
        loss_mask_expanded = tf.broadcast_to(loss_mask[..., tf.newaxis], generated_audio.shape )
        
        # Unscaling
        if self.scaler is not None:
            B, C, L, S = generated_audio.shape
            generated_audio = einops.rearrange(generated_audio, 'b c l s -> (b l s) c')
            audio_expanded = einops.rearrange(audio_expanded, 'b c l s -> (b l s) c')
            
            if isinstance(self.scaler, QuantileTransformer):
                generated_audio = generated_audio.numpy() - 0.05
                audio_expanded = audio_expanded.numpy() - 0.05
                
            generated_audio = self.scaler.inverse_transform( generated_audio )
            audio_expanded = self.scaler.inverse_transform( audio_expanded )
            
            generated_audio = einops.rearrange(generated_audio, '(b l s) c -> b c l s', b=B, l=L, s=S, c=C)
            audio_expanded = einops.rearrange(audio_expanded, '(b l s) c ->  b c l s', b=B, l=L, s=S, c=C)
            
        # Getting Metrics        
        for k in self.map_mname_predmetric:
            
            if getattr(self.map_mname_predmetric[k], 'update_state_non_masked_input', False ):
                self.map_mname_predmetric[k].update_state(audio, generated_audio, loss_mask_expanded, **other )
            else:
                audio_expanded_masked = tf.boolean_mask(audio_expanded, loss_mask_expanded )
                generated_audio_masked = tf.boolean_mask(generated_audio, loss_mask_expanded )
        
                self.map_mname_predmetric[k].update_state( audio_expanded_masked, generated_audio_masked )

        # Preparing output params
        outp = {'cond_mask':cond_mask,
                'loss_mask':loss_mask,
                'target':other.get('target', None),
                'audio':audio,
                'generated_audio':generated_audio
                }
        
        outp = {k:v for k,v in outp.items() if v is not None}

        return outp

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.train_loss_tracker, self.val_loss_tracker, *self.map_mname_trainmetric.values(), *self.map_mname_valmetric.values(), *self.map_mname_predmetric.values() ]
    
    def compile(self, optimizer, mixed_precision, **kwargs):
        
        super(ImputationTrainer, self).compile(**kwargs)
        
        self.optimizer = optimizer
        self.optimizer.jit_compile = False
        self.mixed_precision = mixed_precision
        
        if self.mixed_precision is True:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                        
    @staticmethod
    def train(config_trainer, config_data, config_diffusion, config_model ):
        
        # ==== setting up env
        if config_trainer.mixed_precision is True:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        # ==== setting up data generator modules
        data_generator = MAP_NAME_DSET.get( config_trainer.dataset_name, Dset_V2 )(dataset_name=config_trainer.dataset_name, **vars(config_data) )

        # `Adjusting configs - required due to migration to new training scripts
        config_trainer = config_trainer_param_adjust(config_trainer, data_generator.len_train_unbatched()//config_data.batch_size )
        
        #TODO: set up functionality: if validation file does not exist, then dset_val becomes a random subselection of dset_train, not just the final 0.2 etc
        dset_train = data_generator.get_dset_train( config_data.batch_size, config_data.shuffle_buffer_prop, config_trainer.epochs )
        dset_val = data_generator.get_dset_val( config_data.batch_size_val if config_data.batch_size_val else config_data.batch_size )

        # ==== setting up data diffusion module
        diffusion = MAP_NAME_DIFFUSION[config_trainer.diffusion_method]( **vars(config_diffusion) )

        # ==== setting up neural model
        model = MAP_MNAME_MODEL[config_trainer.model_name]( **vars(config_model) )

        # removing any previous logs
        shutil.rmtree( os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name) , ignore_errors=True)

        # saving configs relating to experiment
        os.makedirs(  os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name, 'configs') )

        yaml.dump(vars(config_trainer), open(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'configs','config_trainer.yaml'),'w' ) ) 
        yaml.dump(vars(config_data), open(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'configs','config_data.yaml'),'w' ) )
        yaml.dump(vars(config_diffusion), open(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'configs','config_diffusion.yaml'),'w' ) )
        yaml.dump(vars(config_model), open(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'configs','config_model.yaml'),'w' ) )

        if hasattr( data_generator, 'scaler'):
            pickle.dump( data_generator.scaler, open(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'scaler.pkl'),'wb' ) )

        # ===== Setting up Trainer Module           
        
        # callbacks
        callbacks = []

        callbacks.append(
            ImputationTrainerCallback(config_trainer.dir_ckpt, config_trainer.exp_name)
        )
        
        callbacks.append(
            ks.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config_trainer.patience,
                verbose=1,
                )
            )
        
        prog_bar = tfa.callbacks.TQDMProgressBar(metrics_format='{name}: {value:.2e}')
        callbacks.append(prog_bar)

        callbacks.append(
            ks.callbacks.TensorBoard(
                log_dir=os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name),
                profile_batch = False, #0 if config_trainer.debugging == False else [2,12],
                write_graph=False
                )
            )
        
        os.makedirs( os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'checkpoint') )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'checkpoint',"ckpt.hdf5"),
                save_freq='epoch',
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=0    
                )
            )
        
        trainer = ImputationTrainer( model=model, diffusion=diffusion, **vars(config_trainer) )
        
        #Optimizer Setup        
        boundaries = [config_trainer.steps_per_epoch ]
        values = [ 1*config_trainer.learning_rate, 1*config_trainer.learning_rate]

        schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        warmup = WarmUp( config_trainer.learning_rate, schedule, config_trainer.steps_per_epoch) # config_trainer.steps_per_epoch//10 )     
        optimizer = ks.optimizers.Adam( warmup, weight_decay=config_trainer.weight_decay )
        

        
        trainer.compile(
            run_eagerly=config_trainer.eager,
            optimizer = optimizer,
            mixed_precision = config_trainer.mixed_precision
        )        
        
        # Building Model
        # dummy_input = ( config_data.batch_size, *dset_train.element_spec.shape[1:])
        # trainer( tf.zeros(dummy_input), tf.ones(dummy_input), False )
                
        # print(trainer.summary())
        
        history = trainer.fit( 
            x=dset_train,
            
            validation_data = dset_val,
            validation_split = None if dset_val else config_trainer.val_split,
            
            workers=config_trainer.workers ,
            epochs= config_trainer.epochs ,
            steps_per_epoch= config_trainer.steps_per_epoch,
            
            validation_steps = config_trainer.validation_steps, 
            callbacks=callbacks,
            verbose=0,
            use_multiprocessing=True
        )

        return None
    
    @staticmethod
    def test(config_trainer, config_data, config_diffusion, config_model):
        # Test should perform the generation and do the testing sampling stuff

        # NOTE: in original code they had no specific way to determine which checkpoint was the one to be used for generation
        # Likely is they tested all of them and choose the best on one the test set. Instead I utilise a validation set methodology
        data_generator = MAP_NAME_DSET.get( config_trainer.dataset_name, Dset_V2 )( dataset_name=config_trainer.dataset_name, **vars(config_data) )
        if hasattr( data_generator, 'scaler'):
            scaler = pickle.load(
                open(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name,'scaler.pkl'),'rb') )
            data_generator.scaler=scaler
            
                
        dset_test = data_generator.get_dset_test(config_data.batch_size_val)

        # diffusion module
        diffusion = MAP_NAME_DIFFUSION[config_trainer.diffusion_method]( **vars(config_diffusion) )

        # model
        model = MAP_MNAME_MODEL[config_trainer.model_name]( **vars(config_model) )    
        
        # Load model state from checkpoint
        trainer = ImputationTrainer( model=model, 
                                    diffusion=diffusion,
                                    scaler=scaler,
                                    **vars(config_trainer) )
        
        if type( dset_test.element_spec ) == dict:
            dummy_input = ( config_data.batch_size_val, *dset_test.element_spec['data'].shape[1:])
        else:
            dummy_input = ( config_data.batch_size_val, *dset_test.element_spec.shape[1:])
            
        trainer.run_eagerly = config_trainer.eager
        trainer( tf.zeros(dummy_input), tf.ones(dummy_input), False )

        # hdf5 save method
        fp_saved_weights =  list( glob.glob( os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name, 'checkpoint' ,'ckpt*' ) ) )[0]
        trainer.load_weights( fp_saved_weights, by_name=True )
        trainer.run_eagerly = config_trainer.eager

        print(trainer.summary())    

        callbacks = [
            # ImputationTrainerCallback
            tfa.callbacks.TQDMProgressBar(metrics_format='{name}: {value:.2e}'),
            ImputationTrainerCallback( config_trainer.dir_ckpt, config_trainer.exp_name)
            ]

        dir_outp = os.path.join(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name, 'test' ))
        os.makedirs(dir_outp, exist_ok=True)
        
        dict_outp = trainer.predict(
            dset_test,
            workers=config_trainer.workers,
            # steps = 1 if config_trainer.debugging else None, 
            # steps = 10, 
            callbacks = callbacks)      
            # use_multiprocessing=False
            # { 'cond_mask': , 'batch':, 'generated_audio':}

        # TODO: Saving Outputs to File    
        with open( os.path.join(dir_outp,'pred_outp.pkl') , 'wb' ) as f:
            pickle.dump( dict_outp, f)

        return True
 
    @staticmethod
    def parse_config(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        # Method versions
        parser.add_argument("--exp_name", default='debug', type=str)
        parser.add_argument("--model_name", default="SSSDS4", choices=["CSDIS4" ,"SSSDSA", "SSSDS4"])
        parser.add_argument("--dataset_name", default="ptbxl_248", choices=[
            'electricity',
            'ettm1_1056',
            'mujoco',
            'ptbxl_248',
            'ptbxl_1000',
            'stocks',
            'stocks_maxmin'])
        parser.add_argument("--diffusion_method", default='alvarez', choices=["alvarez", "csdi"])
        
        # training settings
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--epochs", default=None, type=int)
        
        parser.add_argument("--n_iters", default=150000, type=int )
        parser.add_argument("--steps_per_epoch", default=None, type=float )
        parser.add_argument("--validation_steps", default=None, type=int)
        
        parser.add_argument("--optimizer", default='adam', type=str, choices=['adam','adafactor'])
        parser.add_argument("--mixed_precision", action='store_true')
        parser.add_argument("--eager", action='store_true') 
        parser.add_argument("--grad_accum", default=None, type=int)
        parser.add_argument("--patience", default=20, type=int)
        
        parser.add_argument("--workers", default=10, type=int )
        
        parser.add_argument("--learning_rate", default=1e-4,type=float )
        parser.add_argument("--weight_decay", default=None, type=float)
        parser.add_argument("--val_split", type=float, default = None)


        # directories
        parser.add_argument("--dir_ckpt",type=str, default='experiments')
        
        # parser.add_argument("--segment_length",type=int, default=100) #This is dependent on the model -> move to model config
        # parser.add_argument("--sampling_rate",type=int, default=100) #This is dependent on the model -> move to model config

        # loss and metrics
        parser.add_argument("--base_loss", default='mse', type=str)
        parser.add_argument("--train_metrics", type=str, default=None, 
                    help="str with different metrics seperated by underscore e.g. 'mae_mse' ") # mae, mse, rmse, mre
        parser.add_argument("--val_metrics", type=str, default='mae',
                    help='str with different metrics seperated by underscore') # mae, mse, rmse, mre
        parser.add_argument("--pred_metrics", type=str, default='mae',
                    help='str with different metrics seperated by underscore') # mae, mse, rmse, mre, crps
        
        parser.add_argument("--pred_samples", type=int, default=10,
                    help='Number of Samples to generate per datum') # mae, mse, rmse, mre, crps

        parser.add_argument("--eval_all_timestep",
                    action='store_true', default=False,
                    help="During Test/Validation step produce a loss for all time step")

        # miscallaneous 
        parser.add_argument("--test_only",action='store_true', default=False )
        
        parser.add_argument("--debugging",action='store_true', default=False )

        config_trainer = parser.parse_known_args()[0]

        return config_trainer


if __name__ == "__main__":
    
    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # args trainer
    config_trainer = ImputationTrainer.parse_config(parent_parser)
    
    if config_trainer.test_only==False:
        # add model specific args
        config_model = MAP_MNAME_MODEL[config_trainer.model_name].parse_config(parent_parser)
        
        # add data specific args
        config_data = MAP_NAME_DSET[config_trainer.dataset_name].parse_config(parent_parser)

        # add diffusion specific args
        config_diffusion = MAP_NAME_DIFFUSION[config_trainer.diffusion_method].parse_config(parent_parser)
        
        ImputationTrainer.train(config_trainer, config_data, config_diffusion, config_model )
        ImputationTrainer.test(config_trainer, config_data, config_diffusion, config_model )


    if config_trainer.test_only==True:
        exp_name = config_trainer.exp_name
        dir_ckpt = config_trainer.dir_ckpt

        # NOTE: This setup assumes we use the same settings in train and test
        config_trainer = argparse.Namespace( ** yaml.safe_load(open(os.path.join(dir_ckpt, exp_name,'configs','config_trainer.yaml'),'r' ) ) )
        config_data =  argparse.Namespace( **yaml.safe_load( open(os.path.join(dir_ckpt, exp_name,'configs','config_data.yaml'), 'r' ) ) )
        config_diffusion = argparse.Namespace( **yaml.safe_load(open(os.path.join(dir_ckpt, exp_name,'configs','config_diffusion.yaml'),'r' ) ) )
        config_model = argparse.Namespace( **yaml.safe_load( open(os.path.join(dir_ckpt, exp_name,'configs','config_model.yaml'),'r' ) ) )

        ImputationTrainer.test( config_trainer, config_data, config_diffusion, config_model )
    