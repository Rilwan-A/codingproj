import tensorflow as tf
import argparse
import numpy as np
from functools import lru_cache
import os
import pickle 
import yaml 
import copy 
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class DsetYahooStocks():
    """
    Datasets that contain one file, to be split between train, validation, test
    
    Used for the Stock data from Yahoo

    """
    def __init__(self,
            dir_data,                         
            start_prop_train=0.0,
            end_prop_train=0.8,

            start_prop_val=0.8,
            end_prop_val=1.0,

            start_prop_test=None,
            end_prop_test=None,
            
            window_size=56,
            window_shift=None,
            test_set_method='normal',
            scaler = None,
            window_sample_count =None,
            **kwargs ):
        
        self.data_path = os.path.join( dir_data, 'allstock_tz_ignored.pkl' )
        self.data_path_currency =  os.path.join( dir_data, 'currencies.pkl' ) 
        self.map_index_tickers = yaml.safe_load(open( os.path.join( dir_data, 'dict_indicies.yaml'),"r")) 
        
        
        self.start_prop_train = start_prop_train
        self.end_prop_train = end_prop_train

        self.start_prop_val = start_prop_val
        self.end_prop_val = end_prop_val

        self.start_prop_test = start_prop_test
        self.end_prop_test = end_prop_test
        
        self.window_size = window_size
        self.window_shift = window_size if window_shift is None else window_shift

        self.test_set_method = test_set_method
        
        self.window_sample_count = window_sample_count
        
        
        if scaler is None:
            self.scaler = None
        elif scaler == 'quantile':
            self.scaler = QuantileTransformer( n_quantiles=2000, subsample=10000 ) 
        elif scaler  == 'standard':
            self.scaler = StandardScaler()     
    @lru_cache
    def __len__(self):
        return len(pickle.load( open(self.data_path,"rb")))  // self.window_shift
 
    def len_train_unbatched(self):
        return  int( (self.end_prop_train-self.start_prop_train) * len(self) )

    def len_val_unbatched(self):
        return int( (self.end_prop_val-self.start_prop_val)  * len(self) )
    
    def len_test_unbatched(self):
        return int( (self.end_prop_test-self.start_prop_test)  * len(self) )
    
    def get_dset(self, batch_size=64, shuffle_buffer_prop=0, prefetch=True, cache=False, 
                 s_idx=None, e_idx=None, 
                 indexes_to_include = ['eurostoxx','hangseng','dowjones'],
                 epochs=1):
        
        # Load data
        data = pickle.load( open(self.data_path,"rb"))
        data_currency =  pickle.load( open(self.data_path_currency,"rb"))
        
        # Loading the stock tickers to keep
        indexes_to_include = copy.deepcopy( indexes_to_include )
        tickers_to_keep = sum( (self.map_index_tickers[idx] for idx in indexes_to_include), [] )

        # Selecting price close column
        data = data.xs('Close', level=1, axis=1, drop_level=True)        
        data_currency = data_currency.xs('Close', level=1, axis=1, drop_level=True) 
        
        # Filtering data by tickers to keep'
        data = data[tickers_to_keep]
                
        # Adding currency information
        if 'eurostoxx' not in indexes_to_include:
            data_currency = data_currency.drop('EURUSD=X', axis=1)
        if 'hangseng' not in indexes_to_include:
            data_currency = data_currency.drop('CNYUSD=X', axis=1)


        data = data.join( data_currency )
        
        # date filtering
        if e_idx is not None:
            e_date = data.index[e_idx]
            data = data[ data.index <= e_date]
                    
        if s_idx is not None:
            s_date = data.index[s_idx]
            data = data[ data.index >= s_date]
        
        #Apply log return transformation to price data
        data_price = copy.deepcopy(data[1:])
        data = self.log_return_transform(data)
        
        try:
            data_std = self.scaler.transform(data.values)
            if isinstance(self.scaler, QuantileTransformer):
                data_std = data_std + 0.05
        
        except NotFittedError:
            data_std = self.scaler.fit_transform(data.values)
            if isinstance(self.scaler, QuantileTransformer):
                data_std = data_std + 0.05

        data_std = pd.DataFrame(data_std, index=data.index, columns=data.columns)

        # get mask over days where stock has not started recording data
        mask_pre_record = self.get_mask_pre_record(data_std)

        # get holiday mask 
        mask_holiday = data_std.isna()
        mask_holiday = np.where(mask_pre_record, False, mask_holiday) #Excluding pre record days from holiday mask
        mask_holiday[:, -data_currency.shape[1]: ] = False #Exclude currency holiday days from mask
                                        
        dict_data = {
            'data':data_std.fillna(0.0).values,
            'target_price':data_price.values,
            # 'mask_holiday':mask_holiday,
            # 'mask_pre_record':mask_pre_record
            'observed_mask':np.logical_and(~mask_holiday, ~tf.cast(mask_pre_record, dtype=tf.bool)),
        }
        
        # Filling in Na values       
        dset = tf.data.Dataset.from_tensor_slices(dict_data)

        if cache:
            dset = dset.cache()
            
        # Creating windows over dataset
        def make_window_dataset(ds, window_size=1, shift=1, stride=1):
                        
            windows = ds.window(window_size, shift=shift, stride=stride)

            def sub_to_batch(sub):
                return tf.data.Dataset.zip(  { k:v.batch(window_size, drop_remainder=True) for k,v in sub.items() }  )

            windows = windows.flat_map(sub_to_batch)
            return windows
        
        if epochs == 1:
            dset = make_window_dataset(dset, window_size=self.window_size, shift = self.window_shift, stride=1)

        elif epochs > 1 and shuffle_buffer_prop == 0:
            dset = dset.repeat(epochs)

        else:
            # Creating windows over dataset, 
            # window differ for each epoch
            li_dsets = [None]*epochs
            for idx in range(epochs):
                _ = dset.skip(random.randint(0, self.window_shift-1))
                dset_window = make_window_dataset(_, window_size=self.window_size, shift = self.window_shift, stride=1)
                li_dsets[idx] = dset_window
            
            dset = li_dsets[0]
            for idx in range(1, epochs):
                dset = dset.concatenate(li_dsets[idx])
        
        #shuffle
        if shuffle_buffer_prop>0:
            shuffle_buffer_size = shuffle_buffer_prop*(e_idx-s_idx)/self.window_size
            dset = dset.shuffle(int(shuffle_buffer_size), reshuffle_each_iteration=True)
        
        if prefetch:
            dset = dset.prefetch(tf.data.AUTOTUNE)
        
        dset = dset.batch(batch_size)
        
        dset = dset.map( lambda dict_:  { 
                                           k: tf.transpose(v,(0,2,1) ) 
                                                for k,v in dict_.items() }  
                                            ) 

        return dset
    
    def get_dset_train(self, batch_size, shuffle_buffer_prop=1.0, epochs=1 ):
        
        s_idx = int( self.start_prop_train * len(self) * self.window_shift ) 
        e_idx = int( self.end_prop_train *len(self) * self.window_shift ) 
        
        dset_train = self.get_dset(batch_size, shuffle_buffer_prop=shuffle_buffer_prop,
                                   prefetch=True, cache=True, s_idx=s_idx, e_idx=e_idx,
                                   epochs=epochs )
        
        return dset_train

    def get_dset_val(self, batch_size ):
        
        s_idx = int( self.start_prop_val * len(self) * self.window_shift ) 

        e_idx = int( self.end_prop_val *len(self) * self.window_shift ) 
        
        dset_val = self.get_dset(batch_size, shuffle_buffer_prop=0, s_idx=s_idx, e_idx=e_idx, cache=True)
        
        return dset_val

    def get_dset_test(self, batch_size, indexes_to_include=['eurostoxx','hangseng','dowjones'] ):
        """ Generate test set

            Then
            
        Args:
            batch_size (_type_): _description_
            shuffle_buffer_size (int, optional): _description_. Defaults to 0.
            holidays_only (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        s_idx = self.start_prop_test if self.start_prop_test is None else int( self.start_prop_test * len(self) * self.window_shift )
        e_idx = self.end_prop_test if self.end_prop_test is None else int( self.end_prop_test *len(self) * self.window_shift )
            
        if self.test_set_method == 'holidays_only':
            # Predict for all holidays in whole period
            dset_test = self.get_dset_holidays_only(batch_size, indexes_to_include, s_idx=s_idx, e_idx=e_idx )
        else:            

            dset_test = self.get_dset(batch_size, shuffle_buffer_size=0, s_idx=s_idx, e_idx=e_idx )
        return dset_test
    
    def get_dset_holidays_only(self, batch_size, indexes_to_include, s_idx=None, e_idx=None):
        
        """Return a dictionary of the form 
            {
                'stockindex1':{
                    [
                        {'s_idx':s1, 'len':l1, 'li_w_idx':li_w1, 'li_windows':li_windows, 'li_holiday_mask':li_holiday_mask, 'li_loss_mask':'li_loss_mask' },
                        {'s_idx':s2, 'len':l2, 'li_w_idx':li_w2, 'li_windows':li_windows, 'li_holiday_mask':li_holiday_mask, 'li_loss_mask':'li_loss_mask' },
                    ],
                'stockindex2':{
                    ...
                }
            }
            Returns:
                _type_: _description_
        """        
        # Load data
        data = pickle.load( open(self.data_path, "rb"))
        data_currency =  pickle.load( open(self.data_path_currency, "rb"))
        data = data.xs('Close', level=1, axis=1, drop_level=True)        
        data_currency = data_currency.xs('Close', level=1, axis=1, drop_level=True)       
        # Loading the stock tickers to keep
        indexes_to_include = copy.deepcopy( indexes_to_include )
        
        # Adding currency information to dataframe
        if 'eurostoxx' not in indexes_to_include:
            data_currency = data_currency.drop('EURUSD=X', axis=1)
        if 'hangseng' not in indexes_to_include:
            data_currency = data_currency.drop('CNYUSD=X', axis=1)

        data = data.join( data_currency  )

        # date filtering
        if e_idx is not None:
            e_date = data.index[e_idx]
            data = data[ data.index <= e_date]
            
        if s_idx is not None:
            s_date = data.index[s_idx]
            data = data[ data.index >= s_date]


        # Apply log return transformation to price data
        data_price = copy.deepcopy(data[1:])
        data = self.log_return_transform(data)
        
        data_std = self.scaler.fit_transform(data.values)
        if isinstance(self.scaler, QuantileTransformer):
            data_std = data_std + 0.05
        
        data_std = pd.DataFrame(data_std,
                                index=data_price.index,
                                columns=data_price.columns)
        
    
        # Divide data into each seperate stock indexes
        dict_index_data = { index:data_std.loc[:,self.map_index_tickers[index]] for index in indexes_to_include }
        # For each group of stocks, we gather a list of dictionaries:
            # Each dictionary contains positional information on a particular holiday that group of stocks share    
        # After this for loop dict_index_holidayinfo[index] = [ {'s_idx':s, 'len': }, .. ]
        dict_index_holidayinfo = {} 
        for index in indexes_to_include:
            
            index_data = dict_index_data[index]
            
            # Getting a mask for holiday days
            mask_pre_record = self.get_mask_pre_record(index_data)
            mask_is_na = index_data.isna()
            
            # Getting the starting index of holiday days
                # For stocks from exchange e, they all share the same holiday
                # Therefore holidays are found by gathering days where all stocks have nan value
                # Also we must factor in periods before data was collected for certain stocks
            mask_holiday = np.where(mask_pre_record, False, mask_is_na) #Excluding pre record days from holiday mask
            
            # mask_holiday_ra = mask_holiday.all(axis=1)
            # NOTE: New Rule at least 80% of stocks in an index have no data implies holiday
            mask_holiday_ra =  mask_holiday.sum(1) >= (mask_holiday.shape[1]*0.8)
            
            li_start_end_idxs = []
            
            # Create a list of dicts {'s_idx':s , 'len':l} indicating start and end of holiday
            for idx, (val, prev_val) in enumerate( zip(mask_holiday_ra[1:], mask_holiday_ra), start=1):
                    
                if prev_val == False and val == True:
                    li_start_end_idxs.append( {'s_idx':idx, 'len':None } )
                    continue
                
                elif prev_val == True and val == False:
                    li_start_end_idxs[-1]['len'] = idx - li_start_end_idxs[-1]['s_idx']
                    continue
            li_start_end_idxs = [ d for d in li_start_end_idxs if d['len'] is not None ]
            dict_index_holidayinfo[index] = li_start_end_idxs

            
        # For each holiday we add information about the window around it we will use
        # After this loop dict_index_trainingwindowinfo[index] = [ {'s_idx':s, 'len':l, 'li_w_idx':[w1, w2, ..] },.. ]
            # where li_w_idx is a list of integers indicating the relative position of the start of the window
            # relative to the start of the holiday
            # each list contains multiple starts as we sample n windows around each holiday
        random.seed(10)
        dict_index_trainingwindowinfo = copy.deepcopy(dict_index_holidayinfo)
        for index in indexes_to_include:
            
            for idx in range( len(dict_index_trainingwindowinfo[index] ) ):
                
                holidayinfo = dict_index_trainingwindowinfo[index][idx] # holidayinfo {s_idx:s, len:l}
                                
                # Get the start idx of the window
                s_idx = holidayinfo['s_idx']
                l = holidayinfo['len']

                window_bound_start =  max(0, s_idx+l-self.window_size)
                window_bound_end = min(s_idx+l+1, data_std.shape[0]-self.window_size-1 )
                li_w_idx = random.sample( 
                                         range(  window_bound_start, window_bound_end+1 ),
                                         k = min(self.window_sample_count, window_bound_end+1-window_bound_start ) 
                                         ) 
                dict_index_trainingwindowinfo[index][idx]['li_w_idx'] = li_w_idx
                
        
        
        #For each holiday we create the actual data window and related masks
        # After this loop dict_index_trainingwindowinfo[index] = [ {'s_idx':s, 'len':l, 'li_w_idx':[w1, w2, ..].
        #                                                            'data':[tf.Tensor(),...], 'loss_mask':[tf.Tensor(),...],     
        #                                                              'cond_mask':[tf.Tensor(),...], 'observed_mask':[tf.Tensor(),...], 
        #                                                               target_return:[tf.TensorShape(1 , c),...],                                                          
        # },.. ]
        for index in indexes_to_include:
            for idx in range( len( dict_index_trainingwindowinfo[index] ) ):
                
                trainwindowinfo = dict_index_trainingwindowinfo[index][idx] 

                s_idx = trainwindowinfo['s_idx']
                l = trainwindowinfo['len']
                li_w_idx = trainwindowinfo['li_w_idx']
                # list of windows
                
                li_windows = [ data_std[  w_idx : w_idx+self.window_size ] for w_idx in li_w_idx ]
                                
                
                # For each window, creating loss mask that only
                # focuses on a single holiday period for a specific exchange
                li_loss_mask = []
                for i in range(len(li_windows)):
                    loss_mask =  pd.DataFrame(0.0, columns=li_windows[i].columns, index=li_windows[i].index ) 
                                       
                    w_idx = li_w_idx[i]
                    loss_mask.loc[ s_idx-w_idx:s_idx-w_idx+l+1, self.map_index_tickers[index] ] = 1.0 # -w_idx+l+1 to include 1 day after the holiday
                                        
                    li_loss_mask.append(loss_mask)

                # Here we calculate the true return over the holiday e.g. 
                    # for a holiday from d1 to dn, we want price(d_n+1)/price(d_0)
                    # All     
                data_price_for_holiday = data_price.iloc[s_idx-1:s_idx+l ] # minus/plus 1 since we need price for the day before/after the holiday to calculate return around the holiday
                return_over_holiday = data_price_for_holiday[-1:] / data_price_for_holiday[:1].values
                li_c = [c for c in return_over_holiday if c not in self.map_index_tickers[index] ]
                return_over_holiday.loc[:, li_c ] = 0.0 #np.nan # zeroing out the returns relating to stocks not in the stock index we are iterating over
                
                li_return_over_holiday = [ return_over_holiday ]*self.window_sample_count
                # NOTE: My logic for choosing holidays for a group of stocks is based on all stocks having no data available
                # this logic seems to not work in a handful of cases, e.g. there are days for an exchange whereby only 3 of
                # the stocks have data available. Investigation showed that this phenomenon happened on christmas eve for some reason
                # The issue is that eurostoxx contains, stocks from diff countries who have diff holidays, e.g. Finland seems to have figures on 24th dec
                # Solution: New Rule: If over 90% of stocks have no reading, then declare it a holiday.
                # NOTE: Need to make sure to update the training logic too 

                trainwindowinfo['li_s_idx'] = [s_idx]*len(li_windows)
                trainwindowinfo['li_len'] = [l]*len(li_windows)
                trainwindowinfo['li_data'] = [ w.fillna(0.0).values for w in li_windows]
                trainwindowinfo['li_loss_mask'] = [ lm.values for lm in li_loss_mask]
                trainwindowinfo['li_cond_mask'] = [ (1-mask.values) for mask in li_loss_mask]
                trainwindowinfo['li_observed_mask'] =  [ ~data.isna() for data in li_windows]
                trainwindowinfo['li_target_return'] =  [ r.values for r in li_return_over_holiday]
                
                del trainwindowinfo['s_idx']
                del trainwindowinfo['len']
                     
        # Adding information to each dict about which stock index this eachdatum belongs to
        for index in indexes_to_include:
            for idx in range( len(dict_index_trainingwindowinfo[index] ) ):
                dict_index_trainingwindowinfo[index][idx]['li_index'] = [index]*len(dict_index_trainingwindowinfo[index][idx]['li_data'])
        
        # Flattening dict_index_trainingwindowinfo
            # removing stock index hierarchy
        li_trainingwindowinfo = sum( [ dict_index_trainingwindowinfo[index] for index in indexes_to_include ], [])
            # For each trainindwindowinfo, decompressing the dict of lists structure to a list of dicts
            # e.g. trainingwindowinfo contains info about n windows related to a single holiday, that will be converted to n dicts
        
        li_trainingwindowinfo = [
                [  { k.replace('li_',''):trainindwindowinfo[k][idx] for k in trainindwindowinfo.keys() }  
                    for idx in range(len(trainindwindowinfo['li_data'])) 
                    ]
                for trainindwindowinfo in li_trainingwindowinfo ]
            # flattening groups of related windows
        li_trainingwindowinfo = sum(li_trainingwindowinfo,[])
        
        
        # Stacking values of each key
        l = len(li_trainingwindowinfo)
        keys = li_trainingwindowinfo[0].keys()
        dict_input = {k: np.stack( [li_trainingwindowinfo[idx][k] for idx in range(l) ] ) for k in keys  }
        
        # Transposing relevant inputs
        keys_to_transpose = ['data','loss_mask','cond_mask','observed_mask','target_return']
        for k in keys_to_transpose:
            dict_input[k] = np.transpose(dict_input[k], (0,2,1))

        dataset = tf.data.Dataset.from_tensor_slices( dict_input )
        
        dataset = dataset.batch(batch_size)
        
        return dataset
        
    def log_return_transform(self, data):
        # Apply log return transformation to price data
        # For Monday log return is calculated as the return the log Price on Monday - log Price on the previous Sunday
        data = np.log(data[1:]) - np.log(data.shift(1)[1:] ) 
        return data
        
    def get_mask_pre_record(self, data ):
        # Makes a mark which indicates whether records had begun. E.g. True 
            #explains the value is missing since recording had not started
        
        idxs_data_start = data.notna().values.argmax(axis=0)
        template = np.zeros_like(data.values)
        for idx, idx_data_start in enumerate(idxs_data_start):
            template[:idx_data_start, idx ] = 1
        
        mask_pre_record = template
        return mask_pre_record

    @staticmethod
    def parse_config(parent_parser):
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)

        # scaler
        parser.add_argument("--scaler", type=str, default='quantile', choices=['quantile','standard'] )
        
        # dataset location
        parser.add_argument("--dir_data", type=str, default='./datasets/yahoo_data/')
        parser.add_argument("--window_size", type=int, default=20)
        parser.add_argument("--window_shift", type=int, default=None)
        parser.add_argument("--window_sample_count", type=int, default=1)
               
        
        parser.add_argument("--shuffle_buffer_prop", type=float, 
                            default=1.0,
                            help='proportion of the dataset to hold in shuffle buffer')
        
        # dataset sizes
        parser.add_argument("--start_prop_train", default=0.0, type=float )
        parser.add_argument("--end_prop_train", default=0.8, type=float )
        
        
        parser.add_argument("--start_prop_val", default=0.8, type=float )
        parser.add_argument("--end_prop_val", default=1.0, type=float )
        
        parser.add_argument("--start_prop_test", default=None, type=float )
        parser.add_argument("--end_prop_test", default=None, type=float )
        
        parser.add_argument("--test_set_method", choices = ['normal','holidays_only'], default='normal' )

        parser.add_argument("--batch_size", default=60, type=int)
        parser.add_argument("--batch_size_val", default=None, type=int)

        config_data = parser.parse_known_args()[0]

        return config_data

class DsetYahooStocksMaxMin(DsetYahooStocks):

    def get_dset(self, batch_size=64, shuffle_buffer_prop=0, prefetch=True, cache=False, 
                 s_idx=None, e_idx=None, 
                 indexes_to_include = ['eurostoxx','hangseng','dowjones'],
                 epochs=1):
        
        # Load data
        data = pickle.load( open(self.data_path,"rb"))
        data_currency =  pickle.load( open(self.data_path_currency,"rb"))
        
        # Loading the stock tickers to keep
        indexes_to_include = copy.deepcopy( indexes_to_include )

                
        # Adding currency information
        data = data.join( data_currency )
        
        # date filtering
        if e_idx is not None:
            e_date = data.index[e_idx]
            data = data[ data.index <= e_date]
                    
        if s_idx is not None:
            s_date = data.index[s_idx]
            data = data[ data.index >= s_date]


        # Dropping 'Open' columns
        data = data.drop( [c for c in data.columns if 'Open'==c[1] ], axis=1 )
        
        # Making New Column High-Low
        # NOTE: do this in a more elegant way
        hl_col_labels = [  (c, 'HighLow') for c in data.xs('Close', level=1, axis=1, drop_level=True).columns ]
        data[hl_col_labels] = data.xs('High', level=1, axis=1, drop_level=True).values  - data.xs('Low', level=1, axis=1, drop_level=True).values       
        
        # drop the High and Low columns
        data = data.drop( [c for c in data.columns if c[1] in ['High','Low'] ] , axis=1)
        
        try:
            data_std = self.scaler.transform(data.values)
            if isinstance(self.scaler, QuantileTransformer):
                data_std = data_std + 0.05
        except NotFittedError:
            data_std = self.scaler.fit_transform(data.values) 
            if isinstance(self.scaler, QuantileTransformer):
                data_std = data_std + 0.05
            

        data_std = pd.DataFrame(data_std, index=data.index, columns=data.columns)

        # get mask over days where stock has not started recording data
        mask_pre_record = self.get_mask_pre_record(data_std)

        # get holiday mask 
        mask_holiday = data_std.isna()
        mask_holiday = np.where(mask_pre_record, False, mask_holiday) #Excluding pre record days from holiday mask
        mask_holiday[:, -data_currency.shape[1]: ] = False #Exclude currency holiday days from mask
                             
        dict_data = {
            'data':data_std.fillna(0.0).values,
            # 'target_price':data.values,
            'observed_mask':np.logical_and(~mask_holiday, ~tf.cast(mask_pre_record, dtype=tf.bool)),
            'idx_hilo_hang_seng_stock': [[ idx for idx, col in enumerate(data_std.columns) if (col[0] in self.map_index_tickers['hangseng']) and ( col[1] == 'HighLow' )  ]]*len(data),
            'idx_hang_seng_stock': [ [ idx for idx, col in enumerate(data_std.columns) if (col[0] in self.map_index_tickers['hangseng'])  ] ]*len(data)
        }
        
        # Filling in Na values       
        dset = tf.data.Dataset.from_tensor_slices(dict_data)

        if cache:
            dset = dset.cache()
            
        # Creating windows over dataset
        def make_window_dataset(ds, window_size=1, shift=1, stride=1):                        
            windows = ds.window(window_size, shift=shift, stride=stride)

            def sub_to_batch(sub):
                return tf.data.Dataset.zip(  { k:v.batch(window_size, drop_remainder=True) for k,v in sub.items() }  )

            windows = windows.flat_map(sub_to_batch)
            return windows
        
        if epochs == 1:
            dset = make_window_dataset(dset, window_size=self.window_size, shift = self.window_shift, stride=1)

        elif epochs > 1 and shuffle_buffer_prop == 0:
            dset = dset.repeat(epochs)

        else:
            # Creating windows over dataset, 
            # window differ for each epoch
            li_dsets = [None]*epochs
            for idx in range(epochs):
                _ = dset.skip(random.randint(0, self.window_size-1))
                dset_window = make_window_dataset(_, window_size=self.window_size, shift = self.window_shift, stride=1)
                li_dsets[idx] = dset_window
            
            dset = li_dsets[0]
            for idx in range(1, epochs):
                dset = dset.concatenate(li_dsets[idx])

        # Adding Loss Mask - aim is to only predict last HighLow columns in window for stocks in HangSang Index 
        def get_loss_mask(shape, idx_hilo_hang_seng_stock):
                                                  

            L, C = shape
            
            mask0 = tf.zeros( (L-1, C) , tf.float32)
            
            mask1 = tf.zeros( (C), tf.float32)

            mask1 = tf.tensor_scatter_nd_update(mask1, tf.split(idx_hilo_hang_seng_stock[:1], [1]*idx_hilo_hang_seng_stock.shape[1], axis=1 ) ,
                                                tf.split(tf.ones_like(idx_hilo_hang_seng_stock[0], tf.float32),  [1]*idx_hilo_hang_seng_stock.shape[1] )   )
            
            mask1 = tf.expand_dims(mask1, axis=0)

            loss_mask = tf.concat( [mask0, mask1], axis=0 )
            
                    
            return loss_mask
        
        def get_cond_mask( shape ):
            
            L, C = shape
            
            mask0 = tf.ones( (L-1, C) , tf.float32)
            
            mask1 = tf.zeros( (1,C), tf.float32)

            cond_mask = tf.concat( [mask0, mask1], axis=0 )
           
            
            return cond_mask
            
        dset = dset.map(
            lambda dict_ : dict_ | {
                                        'loss_mask': get_loss_mask( dict_['data'].shape, dict_['idx_hilo_hang_seng_stock'])
                                        # 'loss_mask':tf.py_function(get_loss_mask, [dict_['data'].shape, dict_['idx_hilo_hang_seng_stock']], Tout=tf.float32),
                                         }
        )
        dset = dset.map(
            lambda dict_ : dict_ | {
                                        'cond_mask': get_cond_mask( dict_['data'].shape ),
                                        
                                         }
        )
        
        #shuffle
        if shuffle_buffer_prop>0:
            shuffle_buffer_size = shuffle_buffer_prop*(e_idx-s_idx)/self.window_size
            dset = dset.shuffle(int(shuffle_buffer_size), reshuffle_each_iteration=True)
        
        if prefetch:
            dset = dset.prefetch(tf.data.AUTOTUNE)
        
        dset = dset.batch(batch_size)

        dset = dset.map( lambda dict_:  { 
                                           k: tf.transpose(v,(0,2,1) ) 
                                              for k,v in dict_.items() if k not in ['idx_hilo_hang_seng_stock', 'idx_hang_seng_stock']  }
                                            ) 


        return dset
    
    def get_dset_test(self, batch_size, indexes_to_include=['eurostoxx','hangseng','dowjones'] ):
        """ Generate test set

            Then
            
        Args:
            batch_size (_type_): _description_
            shuffle_buffer_size (int, optional): _description_. Defaults to 0.
            holidays_only (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        s_idx = self.start_prop_test if self.start_prop_test is None else int( self.start_prop_test * len(self) )
        e_idx = self.end_prop_test if self.end_prop_test is None else int( self.end_prop_test *len(self) )
        
        dset_test = self.get_dset(batch_size, shuffle_buffer_prop=0, s_idx=s_idx, e_idx=e_idx )
        return dset_test

class Dset_V2():
    """
    Datasets that contain a seperate file for each of train, validate and test
    Datasets in .npy format
    The masks are not contained in the file

    """
    def __init__(self,
        dataset_name,
        dir_data,
        dataset_shape,
        **kwargs
        ):

        self.data_path_train = os.path.join(dir_data, 'train_'+dataset_name+'.npy')
        self.data_path_val = os.path.join(dir_data, 'val_'+dataset_name+'.npy')
        self.data_path_test = os.path.join(dir_data, 'test_'+dataset_name+'.npy')
        
        self.dataset_shape = dataset_shape

    @lru_cache
    def len_train_unbatched(self):
        return len(np.load(self.data_path_train))

    @lru_cache
    def len_val_unbatched(self):
        return len(np.load(self.data_path_val))
    
    def len_test_unbatched(self):
        return len(np.load(self.data_path_test))

    def get_dset(self, data_path, batch_size=64, shuffle_buffer_prop=0, prefetch=False, cache=False, epochs=1):
        
        if not os.path.exists(data_path):
            return None

        data = np.load(data_path)
        
        data = self.rearrange(data)
        
        dset = tf.data.Dataset.from_tensor_slices(data)
        
        if cache:
            dset = dset.cache()
            
        if shuffle_buffer_prop>0:
            shuffle_buffer_size = int(len(data)*shuffle_buffer_prop)
            dset = dset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        del data
        
        if epochs>1:
            dset = dset.repeat(epochs)

        dset = dset.batch(batch_size)

        if prefetch:
            dset = dset.prefetch(tf.data.experimental.AUTOTUNE)

        return dset
    
    def rearrange(self, data):
        """This class assumes that data on file has shape [D, c, seq] = dataset size, channels, sequence length

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        correct_shape = ['D','c','seq']
        if self.dataset_shape == ['D','c','seq']:
            return data
        
            
        data = np.transpose(data, [ self.dataset_shape.index(dim) for dim in correct_shape ] )
        
        return data
        
    def get_dset_train(self, batch_size, shuffle_buffer_prop=0, epochs=1 ):
        
        dset_train = self.get_dset(self.data_path_train, batch_size, shuffle_buffer_prop=shuffle_buffer_prop, prefetch=True, cache=False, epochs=epochs )
        
        return dset_train

    def get_dset_val(self, batch_size, epochs=1):
        
        dset_val = self.get_dset(self.data_path_val, batch_size, shuffle_buffer_prop=0, prefetch=True, cache=False, epochs=epochs )
        
        return dset_val

    def get_dset_test(self, batch_size ):
        
        dset_test = self.get_dset(self.data_path_test, batch_size, shuffle_buffer_prop=0, prefetch=True, cache=False )

        return dset_test

    @staticmethod
    def parse_config(parent_parser):
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)

        # dataset location
        parser.add_argument("--dir_data", type=str, default='./datasets/alvarez', )
        parser.add_argument("--shuffle_buffer_prop", type=float, default=1.0 , help='proportion of the dataset to hold in shuffle buffer')
        parser.add_argument("--batch_size", default=60, type=int)
        parser.add_argument("--batch_size_val", default=None, type=int)
        parser.add_argument("--dataset_shape", type= lambda dim_str: dim_str.split('_'), default=['D','c','seq'] ,
                                        help="The in file dataset shape. D represents dataset length, c channels, and seq is sequence length. Seperate each dim with underscore  ")

        config_data = parser.parse_known_args()[0]

        return config_data

