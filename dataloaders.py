import numpy as np
import pandas as pd
import torch
import torch.distributions as td
from torch.utils.data import Dataset, Subset

MAP_DISTR_SAMPLE = {
    'uniform': lambda lb, ub: td.Uniform(lb, ub),
    
    'mv_uniform': lambda lb=0, ub=1 : td.Independent( td.Uniform(lb, ub), 1 ),

    'mv_normal': lambda loc=torch.zeros((6,)), covariance_matrix=torch.eye(6): td.MultivariateNormal( loc , covariance_matrix ),
    
    'mv_lognormal': lambda loc=torch.zeros((6,)), scale=torch.ones( (6,) ) : td.Independent( td.LogNormal( loc , scale ), 1 )

}


class ToyDataset(Dataset):
    def __init__(self, features, target ):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.target[idx]
        return feature, target
    
    @staticmethod
    def get_toy_dataset(  input_distribution='uniform',
                                sample_size=1000,
                                inp_sample_params = None,
                                target_func = lambda args: torch.sum(args),
                                noise_method = 'random',
                                noise_sample_params = None,
                                ):

        distr =  MAP_DISTR_SAMPLE[input_distribution](**inp_sample_params)
        X = distr.sample( (sample_size,) )
            #TODO - test situations where more noise is added based on the quantile that Y is in
        X_pertubed = ToyDataset.add_noise(X, noise_method, noise_sample_params) #add noise to X instead of Y
        Y = target_func( X_pertubed )
        if Y.ndim==1:
            Y = Y.unsqueeze(-1)
        # Y = ToyDataset.add_noise(Y, noise_method, noise_sample_params)

        ds = ToyDataset(X,Y)

        train_idx_start = 0
        val_idx_start = int(0.6*sample_size)
        test_idx_start = int(0.8*sample_size)

        # Dividing into train, test, val
        ds_train = Subset(ds, indices = list( range(train_idx_start,val_idx_start) ) ) 
        ds_val = Subset(ds, indices = list( range(val_idx_start, test_idx_start) ))
        ds_test = Subset(ds, indices = list( range(val_idx_start, test_idx_start) ))

        return ds_train, ds_val, ds_test

    @staticmethod
    def add_noise( target,
        method='increasing_at_extremes',
        noise_sample_params=None,
        **kwargs):

        assert method in [ 'random_guassian','increasing_at_extremes', 'increasing_at_maximum', 'increasing_at_minimum' ,'intervals']
        
        if method == 'random_guassian':
            target = target + td.Normal(**noise_sample_params ).sample( tuple(target.shape) ) 
            # target = torch.where( target<0.0, torch.tensor(0.0), target)
            target.clamp_min(0.00)

        # Add noise proportional to decile the data is in
        elif method == 'increasing_at_extremes':
            pass
        # Add relatively more noise to the max deciles
        elif method == 'increasing_at_maximum':
            pass

        # Add relatively more noise to the minimum deciles
        elif method == 'increasing_at_minimum':
            pass

        # Add more noise proportional to the size of the value
        elif method == 'intervals':
            pass
        else:
            raise ValueError
        
        return target