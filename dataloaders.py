from functools import lru_cache
import hashlib
import numpy as np
import pandas as pd
import gc
from pandas._libs import missing
import torch
from torch._C import Value
import torch.distributions as td
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import MinMaxScaler, StandardScaler   
import math
import copy
import os
from sklearn.preprocessing import FunctionTransformer
from torch._six import string_classes
import ujson
import pickle
import regex  as re
from typing import Tuple, Callable,  Union, Dict, List, TypeVar
import argparse
import json
import ujson
import xarray as xr
from netCDF4 import Dataset as nDataset
import itertools as it
import glob
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data import IterableDataset
import random
from utils import tuple_type
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper
from frozendict import frozendict
from collections import OrderedDict
from collections.abc import Collection, Mapping, Hashable
import functools
from itertools import product
"""
    dataloaders.py provides functionality for loading in the following Datasets:

    ToyDataset:
        This is a toy dataset. It can be used to investigate how well a bayesian neural net can model uncertainty.
        A bayesian neural network can learn to model the following two forms of uncertainty:
            1) Uncertainty due to not enough data provided for a particular set of X. e.g. 
                when we have a imbalanced/skewed dataset that has few pairs (X,Y) for some subset of X
            2) Uncertainty due to the dataset being 

    AustraliaDataset
        This a weather dataset. The target is daily rainfall. The input is weather related variables. 
        More info in class description.
    
"""


#classes that allow caching on dictionary

def deep_freeze(thing):
    if thing is None or isinstance(thing, str):
        return thing
    elif isinstance(thing, argparse.Namespace):
        return frozendict({k: deep_freeze(v) for k, v in vars(thing).items()})
    elif isinstance(thing, Mapping):
        return frozendict({k: deep_freeze(v) for k, v in thing.items()})
    elif isinstance(thing, Collection):
        return tuple(deep_freeze(i) for i in thing)
    elif not isinstance(thing, Hashable):
        raise TypeError(f"unfreezable type: '{type(thing)}'")
    else:
        return thing


def deep_freeze_args(func):
    
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*deep_freeze(args), **deep_freeze(kwargs))
    return wrapped

# region -- Era5_Eobs with Time
class Generator():
    """
        Base class for Generator classes
        Example of how to use:
            fn = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
            rain_gen = Generator_rain(fn, all_at_once=True)
            datum = next(iter(grib_gen))
    """

    city_latlon = {
            "London": [51.5074, -0.1278],
            "Cardiff": [51.4816 + 0.15, -3.1791 -0.05], #1st Rainiest
            "Glasgow": [55.8642,  -4.2518], #3rd rainiest
            "Lancaster":[54.466, -2.8007], #2nd hieghest
            "Bradford": [53.7960, -1.7594], #3rd highest
            "Manchester":[53.4808, -2.2426], #15th rainiest
            "Birmingham":[52.4862, -1.8904], #25th
            "Liverpool":[53.4084 , -2.9916 +0.1 ], #18th rainiest
            "Leeds":[ 53.8008, -1.5491 ], #8th
            "Edinburgh": [55.9533, -3.1883],
            "Belfast": [54.5973, -5.9301], #25
            "Dublin": [53.3498, -6.2603],
            "LakeDistrict":[54.4500,-3.100],
            "Newry":[54.1751, -6.3402],
            "Preston":[53.7632, -2.7031 ],
            "Truro":[50.2632, -5.0510],
            "Bangor":[54.2274 - 0, -4.1293 - 0.3],
            "Plymouth":[50.3755 + 0.1, -4.1427],
            "Norwich": [52.6309, 1.2974],
            "StDavids":[51.8812+0.05, -5.2660+0.05] ,
            "Swansea":[51.6214+0.05,-3.9436],
            "Lisburn":[54.5162,-6.058],
            "Salford":[53.4875, -2.2901],
            "Aberdeen":[57.1497,-2.0943-0.05],
            "Stirling":[56.1165, -3.9369],
            "Hull":[53.7676+0.05, 0.3274],
            "Armagh":[54.3503, -6.66528],
            "Bath":[51.380001,-2.360000],
            "Brighton":[50.827778,-0.152778],
            "Cambridge":[52.205276, 0.119167],
            "Canterbury":[51.279999,1.080000],
            "Chelmsford":[51.736099,0.479800],
            "Chester":[53.189999,-2.890000],
            "Coventry":[52.408054, -1.510556],
            "Derby":[52.916668,-1.466667],
            "Exeter":[50.716667,-3.533333],
            "Perth":[56.396999, -3.437000],
            "Sunderland":[54.906101,-1.381130],
            "Wolverhampton":[52.591370,-2.110748],
            "Worcester":[	52.192001,-2.220000],
            "York":[53.958332,-1.080278],
            }
        # The list of boundaries to remove assumes that we are using a 16by16 outer grid with a 4by4 inner grid
    
    invalid_points = { #latitude is the keys, longitude is the values
            #0:9: (0-48)(64-100)
            0:list(range(0,48))+list(range(64,140)), 1:list(range(0,48))+list(range(64,140)), 2:list(range(0,48))+list(range(64,140)), 3:list(range(0,48))+list(range(64,140)), 4:list(range(0,48))+list(range(64,140)), 5:list(range(0,48))+list(range(64,140)), 
            6:list(range(0,48))+list(range(64,140)), 7:list(range(0,48))+list(range(64,140)), 8:list(range(0,48))+list(range(64,140)), 9:list(range(0,48))+list(range(64,140)),

            
            10: list(range(0,48))+list(range(96,140)), 
            
            #11-13: (0,48)(76-140)
            11: list(range(0,48))+list(range(76,140)), 12 :list(range(0,48))+list(range(76,140)), 13 :list(range(0,48))+list(range(76,140)),

            #14-17: (0-48)(96,140)
            14:list(range(0,48))+list(range(96,140)), 15:list(range(0,48))+list(range(96,140)), 16:list(range(0,48))+list(range(96,140)), 17:list(range(0,48))+list(range(96,140)),
            18:list(range(0,48))+list(range(96,140)), 19:list(range(0,48))+list(range(96,140)), 20:list(range(0,48))+list(range(96,140)), 21:list(range(0,48))+list(range(96,140)),
            22:list(range(0,48))+list(range(96,140)), 23:list(range(0,48))+list(range(96,140)), 24:list(range(0,48))+list(range(96,140)), 25:list(range(0,48))+list(range(96,140)),
            26:list(range(0,48))+list(range(96,140)), 27:list(range(0,48))+list(range(96,140)), 28:list(range(0,48))+list(range(96,140)), 29:list(range(0,48))+list(range(96,140)),

            #30-33
            30:list(range(0,48))+list(range(100,140)), 31:list(range(0,48))+list(range(100,140)), 32:list(range(0,48))+list(range(100,140)), 33:list(range(0,48))+list(range(100,140)),


            #38-41: (104-140)
            34:list(range(100,140)), 35:list(range(100,140)), 36:list(range(100,140)), 37:list(range(100,140)),
            38:list(range(104,140)), 39:list(range(104,140)), 40:list(range(104,140)), 41:list(range(104,140)),

            #42-45: (108-140)
            42:list(range(108,140)), 43:list(range(108,140)), 44:list(range(108,140)), 45:list(range(108,140)),

            #46-49: (112-140)
            46:list(range(112,140)), 47:list(range(112,140)), 48:list(range(112,140)), 49:list(range(112,140)),

            #50-61: (120-140)
            50:list(range(120,140)), 51:list(range(120,140)), 48:list(range(120,140)), 49:list(range(120,140)),
            54:list(range(120,140)), 55:list(range(120,140)), 56:list(range(120,140)), 57:list(range(120,140)),
            58:list(range(120,140)), 59:list(range(120,140)), 60:list(range(120,140)), 61:list(range(120,140)),

            #86-100: (0-40)
            86:list(range(0,40)), 87:list(range(0,40)), 88:list(range(0,40)), 89:list(range(0,40)),
            90:list(range(0,40)), 91:list(range(0,40)), 92:list(range(0,40)), 93:list(range(0,40)),
            94:list(range(0,40)), 95:list(range(0,40)), 96:list(range(0,40)), 97:list(range(0,40)), 98:list(range(0,40)), 99:list(range(0,40)), 100:list(range(0,40))

        }
    
    #region
    # creating the grid of all points we don't want to pass to predictor model
    # saves time during training in both data loading and reduced training steps
    s1_1 = list( product( range(0,10), range(0,49) ) ) #0:9: (0-48)
    s1_2_new = list( product( range(0,10), range(78,140) ) )#0:9: (64-140)

    s2_1 = list( product( [10], range(0,49) ) )#10: (0-49)
    s2_2_new = list( product( [10], range(94,140) ) )#10: (96,140)

    s3_1 = list( product( range(11,14), range(0,49) ) )#11-13: (0-49)
    s3_2_new = list( product( range(11,14), range(93,140) ) )#11-13: (76,140)

    s4_1 = list( product( range(14,30), range(0,49) ) )#14,29: (0-49)
    s4_2_new = list( product( range(14,30), range(92,140) ) )#19,26: (96,140)

    s5_1 = list( product( range(30,34), range(0,49) ) )#30,33: (0-49)
    s5_2 = list( product( range(30,34), range(100,140) ) )#30,33: (100,140)

    s6_1_new = list( product( range(34,38), range(0,49) ) )#38,40: (0,49)
    s6_2_new = list( product( range(36,44), range(0,26) ) )#38,40: (0,49)
    s6_3_new = list( product( range(0,100), range(0,10) ) )#38,40: (0,49)
    s6_2_new = list( product( range(32,44), range(102,140) ) )#38,40: (0,49)

    s7_2_new = list( product( range(38,42), range(104,140) ) )#38,41: (104,140)
    s7_3_new = list( product( range(38,46), range(0,24) ) )#38,41: (104,140)

    s7_1 = list( product( range(42,46), range(104,140) ) )#42,45: (108,140)

    s8_1 = list( product( range(46,50), range(112,140) ) )#46,49: (108,140)

    s9_1 = list( product( range(50,62), range(120,140) ) )#50,61: (108,140)

    s10_1_new = list( product( range(74,100), range(0,40) ) )#50,61: (108,140)

    s11_1_new = list( product( range(0,100), range(126,140) ) )#46,49: (108,140)

    s12_1_new = list( product( range(85,100), range(0,50) ) ) + list( product( range(85,100), range(75,140) ) )#46,49: (108,140)

    s12_2_new = list( product( range(90,100), range(50,75) ) )#46,49: (108,140)


    s13_1_new = list( product( range(46,70), range(51,60) ) )#46,49: (108,140)

    s13_2_new = list( product( range(46,54), range(51,73) ) )#46,49: (108,140)

    s13_3_new = list( product( range(68,87), range(40,57) ) )#46,49: (108,140)

    s = s1_1 + s1_2_new + s2_1 + s2_2_new + s3_1 + s3_2_new + s4_1 + s4_2_new + s5_1 + s5_2 +  s6_1_new + s6_2_new + s6_3_new + s7_1 +s7_2_new + s7_3_new + s8_1 + s9_1 + s10_1_new + s11_1_new + s12_1_new + s12_2_new + s13_1_new + s13_2_new +s13_3_new
    invalid_points_vers2 = sorted(list(set(s)))
    
    del s, s1_1 , s1_2_new , s2_1 , s2_2_new , s3_1 , s3_2_new , s4_1 , s4_2_new , s5_1 , s5_2 ,  s6_1_new , s6_2_new , s6_3_new , s7_1 ,s7_2_new , s7_3_new , s8_1 , s9_1 , s10_1_new , s11_1_new , s12_1_new , s12_2_new , s13_1_new , s13_2_new ,s13_3_new
    # endregion
    
    #The longitude lattitude grid for the 0.1 degree E-obs and rainfall data
    latitude_array = np.linspace(58.95, 49.05, 100)
    longitude_array = np.linspace(-10.95, 2.95, 140)

    def __init__(self, fp, lookback=None, iter_chunk_size=None,
                 all_at_once=False, start_idx=0, end_idx=None,
                    dset_start_date=None, dset_time_freq=None, 
                    ):
        """Extendable Class handling the generation of model field and rain data
            from E-Obs and ERA5 datasets

        Args:
            fp (str): Filepath of netCDF4 file containing data.
            all_at_once (bool, optional): Whether or not to load all the data in RAM or not. Defaults to False.
            start_idx (int, optional): Skip the first start_idx elements of the dataset
            
        """ 

        if lookback is not None and iter_chunk_size%lookback!=0:
            print("Iter chunk size must be a multiple of lookback to ensure that the samples we pass to model can be transformed to the correct shape")
            iter_chunk_size = lookback* int( ( iter_chunk_size+  lookback/2)//lookback )

        self.generator = None
        self.all_at_once = all_at_once
        self.fp = fp
        
        # Retrieving information on temporal length of  dataset        
        with nDataset(self.fp, "r") as ds:
            if 'time' in ds.dimensions:
                self.max_data_len = ds.dimensions['time'].size
            else:
                raise NotImplementedError
        
        self.lookback = lookback
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx else self.start_idx + self.max_data_len
        
        self.dset_start_date = dset_start_date
        self.dset_time_freq = dset_time_freq 
        self.date_range = np.asarray( pd.date_range( start= dset_start_date, periods=self.max_data_len, freq=self.dset_time_freq, normalize=True  ) )

        
        # Ensuring we end_idx is a multiple of lookback away 
        if self.lookback != None:
            self.end_idx = self.start_idx+ int(((self.end_idx-self.start_idx)//self.lookback)*self.lookback )

        self.data_len_per_location = self.end_idx - self.start_idx

        self.iter_chunk_size = iter_chunk_size
            
    def yield_all(self):
        pass

    def yield_iter(self):
        pass
    
    def __call__(self, ):
        if(self.all_at_once):
            return self.yield_all()
        else:
            return self.yield_iter()
    
    def find_idxs_of_loc(self, loc="London"):
        """Returns the grid indexes on the 2D map of the UK which correspond to the location (loc) point

        Args:
            loc (str, optional): name of the location. Defaults to "London".

        Returns:
            tuple: Contains indexes (h1,w1) for the location (loc)
        """        
        coordinates = self.city_latlon[loc]
        indexes = self.find_nearest_latitude_longitude( coordinates)  # (1,1)
        return indexes

    def find_idx_of_loc_region(self, loc, dconfig):
        """ Returns the the indexes defining gridded box that surrounds the location of interests

            Raises:
                ValueError: [If the location of interest is too close to the border for evaluation]

            Returns:
                tuple: Returns a tuple ( [upper_h, lower_h], [left_w, right_w] ), defining the grid box that 
                    surrounds the location (loc)
        """
        
        city_idxs = self.find_idxs_of_loc(loc) #[h,w]
        
        # Checking that central region of interest is not too close to the border

        bool_regioncheck_lat = city_idxs[0] > (dconfig.outer_box_dims[0]-dconfig.inner_box_dims[0]) and city_idxs[0] < (city_idxs[0] - (dconfig.outer_box_dims[0]-dconfig.inner_box_dims[0]))
        bool_regioncheck_lon = city_idxs[1] > (dconfig.outer_box_dims[1]-dconfig.inner_box_dims[1]) and city_idxs[1] < (city_idxs[1] - (dconfig.outer_box_dims[1]-dconfig.inner_box_dims[1]))

        # if bool_regioncheck1.any() or bool_regioncheck2.any(): raise ValueError("The specified region is too close to the border")
        if bool_regioncheck_lat or bool_regioncheck_lon: raise ValueError("The specified region is too close to the border")


        # Defining the span, in all directions, from the central region
        if( dconfig.outer_box_dims[0]%2 == 0 ):
            h_up_span = dconfig.outer_box_dims[0]//2 
            h_down_span = h_up_span
        else:
            h_up_span = dconfig.outer_box_dims[0]//2
            h_down_span = dconfig.outer_box_dims[0]//2 + 1

        if( dconfig.outer_box_dims[1]%2 == 0 ):
            w_left_span = dconfig.outer_box_dims[1]//2 
            w_right_span = w_left_span
        else:
            w_left_span = dconfig.outer_box_dims[1]//2
            w_right_span = dconfig.outer_box_dims[1]//2 + 1
        
        #Defining outer_boundaries
        upper_h = city_idxs[0] - h_up_span
        lower_h = city_idxs[0] + h_down_span

        left_w = city_idxs[1] - w_left_span
        right_w = city_idxs[1] + w_right_span
        
        return ( [upper_h, lower_h], [left_w, right_w] )

    def find_nearest_latitude_longitude(self, lat_lon):
        """Given specific lat_lon, this method finds the closest long/lat points on the
            0.1degree grid our input/target data is defined on

        Args:
            lat_lon (tuple): tuple containing the lat and lon values of interest

        Returns:
            tuple: tuple containing the idx_h and idx_w values that detail the posiiton on lat_lon on the 
                0.1degree grid on which the ERA5 and E-Obvs data is defined
        """        
        latitude_index =    np.abs(self.latitude_array - lat_lon[0] ).argmin()
        longitude_index =   np.abs(self.longitude_array - lat_lon[1]).argmin()

        return (latitude_index, longitude_index)
    
    @staticmethod
    @deep_freeze_args
    @lru_cache    
    def get_locs_for_whole_map(dconfig):
        """This function returns a list of boundaries which can be used to extract all patches
            from the 2D map. 

            Args:
                region_grid_params (dictionary): a dictioary containing information on the sizes of 
                    patches to be extract from the main image

            Returns:
                list:return a list of of tuples defining the boundaries of the region
                        of the form [ ([upper_h, lower_h]. [left_w, right_w]), ... ]
            If we do have a stride then shuffle changes which set of points we pick at each round of iteration
        """      
        dconfig = argparse.Namespace(**dconfig)
        original_uk_dim = dconfig.original_uk_dim
        h_shift = dconfig.vertical_shift
        w_shift = dconfig.horizontal_shift
        h_span, w_span = dconfig.outer_box_dims

        #list of values for upper_h and lower_h
        h_start_idx = 0
        range_h = np.arange(h_start_idx, original_uk_dim[0]-h_span, step=h_shift, dtype=np.int32 ) 
        # list of pairs of values (upper_h, lower_h)
        li_range_h_pairs = [ [range_h[i], range_h[i]+h_span] for i in range(0,len(range_h))]
        
        #list of values for (left_w and right_w)
        w_start_idx = 0
        range_w = np.arange(w_start_idx, original_uk_dim[1]-w_span, step=w_shift, dtype=np.int32)
        
        # list of pairs of values (left_w, right_w)
        li_range_w_pairs = [ [range_w[i], range_w[i]+w_span ] for i in range(0,len(range_w))]

        li_boundaries = list( it.product( li_range_h_pairs, li_range_w_pairs ) ) #[ ([h1,h2],[w1,w2]), ... ]
        
        filtered_boundaries = Generator.get_filtered_boundaries( li_boundaries, dconfig.inner_box_dims)

        return filtered_boundaries
    
    @staticmethod
    @deep_freeze_args
    @lru_cache
    def get_locs_latlon_for_whole_map(dconfig):
        dconfig = argparse.Namespace(**dconfig)
        filtered_boundaries = Generator.get_locs_for_whole_map(dconfig)

        filtered_latlon = [
            ( 
                [
                    Generator.hidx_to_lat(hpair_wpair[0][0]),
                    Generator.hidx_to_lat(hpair_wpair[0][1])
                ],
                [
                    Generator.widx_to_lon(hpair_wpair[1][0]),
                    Generator.widx_to_lon(hpair_wpair[1][1])
                ]
            )
            for hpair_wpair in filtered_boundaries
        ]

        return filtered_latlon
    
    @staticmethod
    def hidx_to_lat(hidx):
        return Generator.latitude_array[hidx]
    @staticmethod
    def widx_to_lon(widx):
        return Generator.longitude_array[widx]

    @staticmethod
    def get_filtered_boundaries(li_boundaries, inner_box_dims):

        filtered_boundaries = []

        # Each training datum is defined by its boundaries on a  100by140 grid
        # From this input boundary we predict/output a central region in the boundary
        # We drop datums for which the training datum's output region is more than 85% water

        h_span, w_span = li_boundaries[0]

        h_span_radius = hsr = abs(h_span[0] - h_span[1])//2
        w_span_radius = wsr = abs(w_span[0] - w_span[1])//2


        inner_box_dims_h_radius = ihr = inner_box_dims[0]//2
        inner_box_dims_w_radius = iwr = inner_box_dims[1]//2

        for h_span, w_span in li_boundaries:
            
            mid_point_h = h_span[0] + hsr
            mid_point_w = w_span[0] + wsr
            
            central_h_span = ch_span = [ mid_point_h-ihr, mid_point_h+ihr ]
            central_w_span = cw_span = [ mid_point_w-iwr, mid_point_w+iwr ]

            #gettingwspans for the prediction output region            
            grid_points_in_boundary = list( product( list(range(ch_span[0], ch_span[1]+1)) , list(range(cw_span[0], cw_span[1]+1)) )  )

            count_datum_grid_points = len(grid_points_in_boundary)
            count_datum_invalid_grid_points  = len( [point for point in grid_points_in_boundary if point in Generator.invalid_points_vers2] )

            invalid_coverage = count_datum_invalid_grid_points / count_datum_grid_points

            if invalid_coverage < (1-0.55):
                filtered_boundaries.append((h_span, w_span))

        return filtered_boundaries     

class Generator_rain(Generator):
    """ A generator for E-obs 0.1 degree rain data
    
    Returns:
        A python generator for the rain data
        
    """
    def __init__(self,**generator_params ):
        super(Generator_rain, self).__init__(**generator_params)
        
    def yield_all(self):
        """ Return all data at once
        """
        raise NotImplementedError
        with Dataset(self.fp, "r", format="NETCDF4",keepweakref=True) as ds:
            _data = ds.variables['rr'][:]
            yield np.ma.getdata(_data), np.ma.getmask(_data)   
            
    def yield_iter(self):
        """ Return data in chunks"""

        # xr_gn = xr.open_dataset( self.fp, cache=False, decode_times=False, decode_cf=False, chunks={'time': self.iter_chunk_size})
        with xr.open_dataset( self.fp, cache=True, decode_times=False, decode_cf=False ) as xr_gn:
            idx = copy.deepcopy(self.start_idx)
            # final_idx =  min( self.start_idx+self.data_len_per_location, self.end_idx) 
            # # Same affect as drop_last = True. Ensures that we extract segments with size a mulitple of lookback
            # final_idx = int( self.start_idx + ((final_idx)//self.lookback)*self.lookback )
            # self.data_len_per_location = (final_idx- self.start_idx)
            
            #TODO: Since we are doing strides of lookback length,
            # then add a shuffle ability where we adjust the start idx by up to lookback length so the idx is moved back by at most lookback
            while idx < self.end_idx:

                adj_iter_chunk_size = min(self.iter_chunk_size, (self.end_idx-idx) )
            
                slice_t = slice( idx , idx+adj_iter_chunk_size )
                slice_h = slice( None , None, -1 )
                slice_w = slice( None, None )

                date_windows  = np.asarray( self.date_range[slice_t] )
                marray = xr_gn.isel(time=slice_t ,latitude=slice_h, longitude=slice_w)['rr'].to_masked_array(copy=True)
                array, mask = np.ma.getdata(marray), np.ma.getmask(marray)

                mask = (array==9.969209968386869e+36)
                
                idx+=adj_iter_chunk_size

                yield array, mask, date_windows

    def __call__(self):
        return self.yield_iter()
    
    __iter__ = yield_iter
    
class Generator_mf(Generator):
    """Creates a generator for the model_fields_dataset
    """

    def __init__(self, vars_for_feature=None, **generator_params):
        """[summary]

        Args:
            generator_params:list of params to pass to base Generator class
        """        
        super(Generator_mf, self).__init__(**generator_params)

        self.vars_for_feature = vars_for_feature if vars_for_feature else ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]       
        # self.start_idx = 0
        # self.end_idx =0 
        #self.ds = Dataset(self.fp, "r", format="NETCDF4")
        
        
    def yield_all(self):
        
        xr_gn = xr.open_dataset(self.fp, cache=False, decode_times=False, decode_cf=False)

        slice_t = slice( self.start_idx , self.end_idx )
        slice_h = slice(1,103-2 )
        slice_w = slice(2,144-2)
        
        xr_gn =  xr_gn.isel(time=slice_t, latitude=slice_h, longitude=slice_w)
        
        return xr_gn

    def yield_iter(self):
        # xr_gn = xr.open_dataset(self.fp, cache=False, decode_times=False, decode_cf=False, cache=False, chunks={'time': self.iter_chunk_size})
        with xr.open_dataset(self.fp, cache=True, decode_times=False, decode_cf=False) as xr_gn:
            idx = copy.deepcopy(self.start_idx)

            while idx < self.end_idx:

                adj_iter_chunk_size = min(self.iter_chunk_size, self.end_idx-idx )

                _slice = slice( idx , idx  + adj_iter_chunk_size)
                xr_gen_slice = xr_gn.isel(time=_slice)
                next_marray = [ xr_gen_slice[name].to_masked_array(copy=True) for name in self.vars_for_feature  ]
                
                list_datamask = [(np.ma.getdata(_mar), np.ma.getmask(_mar)) for _mar in next_marray]
                
                _data, _masks = list(zip(*list_datamask))
                # _masks = [ np.logical_not(_mask_val) for _mask_val in _masks] 
                stacked_data = np.stack(_data, axis=-1)
                stacked_masks = np.stack(_masks, axis=-1)

                idx += adj_iter_chunk_size
                
                yield stacked_data[ :, 1:-2, 2:-2, :], stacked_masks[ :, 1:-2 , 2:-2, :] #(100,140,6) 

    __iter__ = yield_iter 


class Era5EobsDataset(IterableDataset):

    def __init__(self, dconfig, start_date, end_date, 
        locations, loc_count, scaler_features=None, 
        scaler_target=None, target_range=None, workers=1, shuffle=False, cache_gen_size=4, **kwargs ) -> None:
        super(Era5EobsDataset).__init__()

        assert target_range is not None or scaler_target is not None
        
        self.dconfig = dconfig
        
        self.locations = locations if locations else dconfig.locations
        
        self.loc_count = loc_count
        
        self.start_date = start_date
        self.end_date = end_date
        start_idx_feat, start_idx_tar = self.get_idx(start_date)
        end_idx_feat, end_idx_tar = self.get_idx(end_date)
        self.gen_size = kwargs.get('gen_size',40)
        self.workers = workers
        self.shuffle = shuffle
        self.cache_gen_size = cache_gen_size
        self.xarray_decode = kwargs.get('xarray_decode', False)
        
        # region Checking for pre-existing scalers - If not exists we create in next section
        self.scaler_features = scaler_features
        
        self.scaler_target = scaler_target 
        
        self.target_range = target_range if target_range is not None else tuple(scaler_target.feature_range)
        
        # endregion
        
        # region Checking for cache of dataset 
        premade_dset_path = os.path.join(dconfig.data_dir,'premade_dset_record.txt')
        if os.path.exists(premade_dset_path):
            premade_dsets = pd.read_csv( premade_dset_path)
        else:
            premade_dsets = pd.DataFrame( columns=['cache_path','scaler_path','start_date','end_date','locations','lookback',
                                                    'target_range','outer_box_dims'] )

        # Query for if cache for dataset and normalizer
        query_res = premade_dsets.query( f"start_date == '{start_date}' and end_date == '{end_date}' and \
                                    locations == '{Era5EobsDataset.locations_enc(self.locations, dconfig=self.dconfig)}' and \
                                    lookback == {str(self.dconfig.lookback_target)} and \
                                    target_range == '{','.join(map(str,self.target_range))}' and \
                                    outer_box_dims == '{','.join(map(str,dconfig.outer_box_dims))}'")
        
        #If it exists we just load it
        if len(query_res)!=0:
            
            self.cache_path = query_res['cache_path'].iloc[0]
            self.cache_exists = os.path.exists(self.cache_path)
            os.makedirs( os.path.join(dconfig.data_dir, "cache"), exist_ok=True )
            # Otherwise set up parameters to use the normal dataset
        else:
            os.makedirs( os.path.join(dconfig.data_dir, "cache"), exist_ok=True )
            self.cache_path = os.path.join( self.dconfig.data_dir, "cache",
                f"start-{start_date}-end-{end_date}_lbtarget-{str(self.dconfig.lookback_target)}-tgtrange-{','.join(map(str,self.target_range))}-outer_box_dims-{','.join(map(str,dconfig.outer_box_dims))}-locs-{Era5EobsDataset.locations_enc(self.locations, dconfig=self.dconfig)}")
            self.cache_exists = os.path.exists(self.cache_path)
        # endregion

        #Finally if any of cache or scalers don't exists we create the non existing ones
        if not self.cache_exists or self.scaler_features == None or self.scaler_target ==  None:
            # Create python generator for rain data
            fp_rain = self.dconfig.data_dir+"/"+self.dconfig.rain_fn
            self.rain_data = Generator_rain(fp=fp_rain,
                                            dset_start_date=self.dconfig.target_start_date,
                                            dset_time_freq = 'D',
                                            all_at_once=False,
                                            iter_chunk_size=self.dconfig.lookback_target*self.gen_size,
                                            lookback=self.dconfig.lookback_target,
                                            start_idx=start_idx_tar,
                                            end_idx = end_idx_tar )

            # Create python generator for model field data 
            mf_fp = self.dconfig.data_dir + "/" + self.dconfig.mf_fn
            self.mf_data = Generator_mf(fp=mf_fp, vars_for_feature=self.dconfig.vars_for_feature, 
                                        dset_start_date=self.dconfig.feature_start_date,
                                        dset_time_freq = '6h',
                                        all_at_once=False, 
                                        iter_chunk_size=self.dconfig.lookback_feature*self.gen_size,
                                        lookback=self.dconfig.lookback_feature,
                                        start_idx=start_idx_feat,
                                        end_idx = end_idx_feat)
            
            self.create_cache_scaler()
        
        self.create_cache_params()
            
        if not self.cache_exists or len(query_res)==0:
            premade_dsets = premade_dsets.append( {
                'cache_path':self.cache_path,
                'start_date':start_date,
                'end_date':end_date,
                'locations':  Era5EobsDataset.locations_enc(self.locations, dconfig=self.dconfig), # '_'.join([loc[:2] for loc in self.locations]),
                'lookback':str(self.dconfig.lookback_target),
                'target_range':','.join(map(str,self.target_range)),
                'outer_box_dims':','.join(map(str,dconfig.outer_box_dims) )
            }, ignore_index=True)
            
            premade_dsets.to_csv( premade_dset_path, index=False)
                   
        # Create buffers for scaler params
        self.features_scale = torch.as_tensor( self.scaler_features.scale_ )
        self.features_mean = torch.as_tensor( self.scaler_features.mean_ )
        self.target_scale = torch.as_tensor( self.scaler_target.scale_ )

    @staticmethod
    def locations_enc(li_locations, **kwargs):
        
        if isinstance( li_locations[0], string_classes):
            # Use string shortening encoding method
            location_encoded = '_'.join([loc[:2] for loc in li_locations])
        
        elif li_locations[0] in Generator.get_locs_for_whole_map(kwargs['dconfig'] ):
            # Use hashing to created shortened string
            
            s = str_li_locations = str(li_locations)
            s = s.encode('utf-8')
            
            location_encoded = hashlib.sha1(s).hexdigest()
        
        return location_encoded
    
    @staticmethod
    def get_dataset( dconfig, target_range, target_distribution_name, **kwargs):

        ds_train = Era5EobsDataset( start_date = dconfig.train_start, end_date=dconfig.train_end,
                                    target_distribution_name=target_distribution_name,
                                    cache_gen_size=dconfig.cache_gen_size,
                                    locations=dconfig.locations,
                                    loc_count=dconfig.loc_count,
                                    target_range=target_range, dconfig=dconfig, 
                                    shuffle=dconfig.shuffle,**kwargs )
        
        ds_val = Era5EobsDataset( start_date = dconfig.val_start, end_date=dconfig.val_end,
                                    dconfig=dconfig,
                                    cache_gen_size=dconfig.cache_gen_size,
                                    target_range=target_range,
                                    target_distribution_name=target_distribution_name,
                                    locations=dconfig.locations,
                                    loc_count=dconfig.loc_count,
                                    scaler_features = ds_train.scaler_features,
                                    scaler_target = ds_train.scaler_target,
                                    shuffle=False,
                                    **kwargs)
        
        
        ds_test = Era5EobsDataset( start_date=dconfig.test_start, end_date=dconfig.test_end,
                                    locations=dconfig.locations_test, loc_count=dconfig.loc_count_test,
                                    cache_gen_size=dconfig.cache_gen_size_test,
                                    dconfig=dconfig,
                                    target_range=target_range,
                                    target_distribution_name=target_distribution_name,
                                    scaler_features = ds_train.scaler_features,
                                    scaler_target = ds_train.scaler_target,
                                    shuffle=False,
                                    xarray_decode=True
                                    )

        return ds_train, ds_val, ds_test, ds_train.scaler_features, ds_train.scaler_target
    
    @staticmethod
    def get_test_dataset( dconfig, target_distribution_name, target_range=None , scaler_features=None, scaler_target=None, **kwargs):


        ds_test = Era5EobsDataset( start_date=dconfig.test_start,
                                    end_date=dconfig.test_end,
                                    locations=dconfig.locations_test, 
                                    loc_count=dconfig.loc_count_test,
                                    dconfig=dconfig,
                                    cache_gen_size=dconfig.cache_gen_size_test,
                                    target_range=target_range,
                                    target_distribution_name=target_distribution_name,
                                    scaler_features = scaler_features,
                                    scaler_target = scaler_target,
                                    xarray_decode=True,
                                    shuffle=False,
                                    **kwargs)

        return  ds_test

    def __iter__(self):
        
        if self.cache_exists and self.scaler_features and self.scaler_target:
                        
            with xr.open_dataset( self.cache_path, decode_cf=self.xarray_decode, decode_times=self.xarray_decode,  cache=True )\
                if os.path.isfile(self.cache_path) else \
                    xr.open_mfdataset( sorted(list( glob.glob(self.cache_path+"/*"))), 
                                      decode_cf=self.xarray_decode, 
                                      decode_times=self.xarray_decode,  
                                        concat_dim='sample_idx',    
                                        combine='nested',     
                                        #coords=['sample_idx'],                             
                                        cache=True ) as xr_cache:
                                   
                if not hasattr(self, 'cache_len'):
                    # for feature, target, target_mask, idx_loc_in_region in self.cached_data:
                    self.cache_len = xr_cache.dims['sample_idx']
                    
                # Making sure its a multiple of lookback away from start idx
                if not hasattr(self, 'cache_start_idx'):
                    self.cache_start_idx = 0

                if not hasattr(self, 'cache_end_idx'):
                    self.cache_end_idx = self.cache_start_idx + self.cache_len
                         
                # Implementing more effecient shuffling method - base on choosing slices to extract at random 
                # replaces holding buffer
                
                lst = li_slices = [ slice( idx , min( idx  + self.cache_gen_size, self.cache_end_idx) ) for idx in range(self.cache_start_idx, self.cache_end_idx, self.cache_gen_size ) ]
                
                if self.shuffle == True:
                    random.shuffle(lst)
                    
                # for _slice in li_slices:
                for s in lst:
                    
                    # ======= xarray method
                    if  isinstance(xr_cache, xr.Dataset):
                    # xarray method
                        xr_cache_slice = xr_cache.isel(sample_idx=s).load()
                        
                        dict_data = { name:torch.tensor( xr_cache_slice[name].data )
                                        for name in ['input','target','mask','idx_loc_in_region'] }

                        dict_data['li_locations'] = xr_cache_slice['li_locations'].data.tolist()
                        try:
                            dict_data['target_date_window'] = xr_cache_slice['target_date_window'].data
                        except ValueError as e:
                            dict_data['target_date_window'] = np.asarray(pd.date_range('1979', periods=0)).reshape((-1,7))
                    
                    # ======== Netcdf4 method   
                    elif isinstance(xr_cache, nDataset):            
                        dict_data = {}
                        
                        dict_data['target'] = xr_cache['target'][s.start:s.end].data
                        dict_data['mask'] = xr_cache['mask'][s.start:s.end].data
                        dict_data['idx_loc_in_region'] = xr_cache['idx_loc_in_region'][s.start:s.end].data
                        dict_data['input'] = xr_cache['input'][s.start:s.end].data
                        try:
                            dict_data['target_date_window'] = xr_cache['target_date_window'][s.start:s.end].data
                        except Exception as e:
                            dict_data['target_date_window'] = np.asarray(pd.date_range('1979', periods=0)).reshape((-1,7))
                        dict_data['li_locations'] = xr_cache['li_locations'][s.start:s.end]
                                                
                        dict_data['target'] = torch.tensor( dict_data['target'] )
                        dict_data['mask'] = torch.tensor( dict_data['mask'] )
                        dict_data['idx_loc_in_region'] = torch.tensor( dict_data['idx_loc_in_region'] )
                        dict_data['input'] = torch.tensor( dict_data['input'] )
                        dict_data['li_locations'] = dict_data['li_locations'].tolist() 
                     
                    if dict_data['target_date_window'].dtype != '<M8[ns]':
                        dict_data['target_date_window'] = dict_data['target_date_window'].astype('<M8[ns]')
                        
                    dict_data['target_date_window'] = dict_data['target_date_window'].astype('datetime64[us]').tolist()
                    
                    # scaling output
                    dict_data['input'] = (dict_data['input'] - self.features_mean)/self.features_scale
                    dict_data['target'] = dict_data['target']*self.target_scale
                    
                    #scaling
                    dict_data['input'] = dict_data['input'].to(torch.float32).squeeze(-1)
                    dict_data['target'] = dict_data['target'].to(torch.float32).squeeze(-1)
                    dict_data['mask'] = dict_data['mask'].to(torch.bool).squeeze(-1)
                                            
                    li_dicts = [ {key:dict_data[key][i:i+1] if key not in ['li_locations','target_date_window'] else dict_data[key][i]
                                  for key in dict_data.keys()} for i in range(len(dict_data['target'])) ]
                                                                    
                    # shuffling data 
                    # unbundling weekly batches and shifting data forward n days 
                    # such that the weekly periods are different
                    if self.shuffle:
                        # We adjust the start idx of model field and rain data
                        # The amount to increment the target, e.g. number of days 
                        target_sub_idx_increment = random.randint(0, self.dconfig.lookback_target)
                        feature_sub_idx_increment = target_sub_idx_increment*4
                        if target_sub_idx_increment>0:
                            li_dicts = self.cache_shuffle(li_dicts, target_sub_idx_increment, feature_sub_idx_increment )

                    # Filter li_dicts that do not have any valid values
                    li_dicts = [ dict_ for dict_ in li_dicts if dict_['mask'].logical_not().any()]
                    
                    yield from li_dicts
                        
        else: 
            print("Note: User is using non cached dataset. The data will not be normalized automatically")
            sample_idx = 0
            # We calculate the maximum number of samples as the start_idx - end_idx // self.lookback for each generator
            
            if self.scaler_features == None: 
                self.scaler_features = StandardScaler()
                bool_update_scaler_features = True
            else:
                bool_update_scaler_features = False

            if self.scaler_target == None: 
                self.scaler_target = MinMaxScaler(feature_range=self.target_range)
                bool_update_scaler_target = True
            else:
                bool_update_scaler_target = False
            
            #Developing Data Cache and Scalers
            
            for idx, ( (feature, feature_mask), (target, target_mask, target_date_window) ) in enumerate(zip(self.mf_data, self.rain_data)):

                dict_data = self.preprocess_batch(feature, feature_mask, target, target_mask, target_date_window, normalize=False)

                if not self.cache_exists:
                    kwargs ={ 
                        "data_vars":{
                                "input":( ("sample_idx","lookback_feat","h_feat","w_feat","d"), torch.concat(dict_data['input']).numpy() ),
                                "target": ( ("sample_idx","lookback_target","h_target","w_target"),torch.concat(dict_data['target']).numpy() ),
                                "mask": ( ("sample_idx","lookback_target","h_target","w_target"),torch.concat(dict_data['mask']).numpy() ),
                                "idx_loc_in_region": ( ("sample_idx","h_w"), np.concatenate(dict_data['idx_loc_in_region']) ) ,
                                "li_locations": ( ("sample_idx",'lookback_target'), np.asarray(sum( dict_data['li_locations'], [])) ),
                                "target_date_window":( ("sample_idx",'lookback_target'), np.concatenate( dict_data['target_date_window'] ) )
                                }
                            }
                                         
                    if idx==0:
                        
                        coords = {
                            "sample_idx": np.arange( torch.concat(dict_data['input']).shape[0] ),
                            "lookback_feat": np.arange( self.dconfig.lookback_feature),
                            "lookback_target": np.arange( self.dconfig.lookback_target),
                            "h_feat": np.arange( dict_data['input'][0].shape[-3]),
                            "w_feat": np.arange( dict_data['input'][0].shape[-2]),
                            "d": np.arange( dict_data['input'][0].shape[-1]),
                            "h_target": np.arange( dict_data['target'][0].shape[-2]),
                            "w_target": np.arange( dict_data['target'][0].shape[-1]),
                            "h_w": np.arange( dict_data['idx_loc_in_region'][0].shape[-1] )
                            }
                        
                        kwargs['coords'] = coords
                        
                        #pass
                        xr_curr = xr.Dataset( **kwargs )
                            
                        if self.dconfig.data_load_method == 'xarray_mult_files_on_disk':
                            #make the folder
                            os.makedirs(self.cache_path, exist_ok=True)
                            curr_f_count = len(list(glob.glob( os.path.join(self.cache_path,"*") )))
                            next_cache_fp = os.path.join( self.cache_path, "{:03d}".format(curr_f_count)+".nc" )
                            # save to file
                            comp = dict(zlib=True, complevel=9)
                            encoding = {var: comp for var in xr_curr.data_vars if (True or var in ['input']) }
                            xr_curr.to_netcdf(next_cache_fp, mode='w', encoding=encoding, unlimited_dims=['sample_idx'] )
                            
                        elif self.dconfig.data_load_method in ['netcdf4_single_file_on_disk']:
                        
                            # Preventing Memory issues in case the file gets to big
                            # Append to a netcd4 file on disk                                
                            nd4_ds = nDataset(self.cache_path, 'w', keepweakref=True )
                            
                            # Creating dimensions
                            for dim_name, dim_value  in kwargs['coords'].items():
                                if dim_name == 'sample_idx':
                                    nd4_ds.createDimension( dim_name,  None )    
                                else:
                                    nd4_ds.createDimension( dim_name,  dim_value.shape[0] )
                            
                            # Inserting current values
                            for val_name, tuple_shape_val in kwargs['data_vars'].items():
                                
                                arr = tuple_shape_val[1]
                                if arr.dtype in [np.bool]:
                                    arr = tuple_shape_val[1].astype(np.int16)
                                if arr.dtype in ['<M8[ns]']:
                                    arr = arr.astype(np.long)

                                if True or (val_name  in []): # ['input']:
                                    nd4_ds.createVariable(val_name, arr.dtype, tuple_shape_val[0], zlib=True, complevel=4, shuffle=False  )
                                else:
                                    nd4_ds.createVariable(val_name, arr.dtype, tuple_shape_val[0] )
                                
                                nd4_ds.variables[val_name][:] = copy.deepcopy(arr)
                                                                                               
                    elif idx!=0:
                                                
                        if self.dconfig.data_load_method == 'xarray_mult_files_on_disk':
                            kwargs['coords'] = {
                                "sample_idx": np.arange( int(xr_curr.sample_idx[-1].data), int(xr_curr.sample_idx[-1].data)+torch.concat(dict_data['input']).shape[0]),

                                "lookback_feat": np.arange( self.dconfig.lookback_feature),
                                "lookback_target": np.arange( self.dconfig.lookback_target),

                                "h_feat": np.arange( dict_data['input'][0].shape[-3]),
                                "w_feat": np.arange( dict_data['input'][0].shape[-2]),

                                "h_target": np.arange( dict_data['target'][0].shape[-2]),
                                "w_target": np.arange( dict_data['target'][0].shape[-1]),
                                "h_w": np.arange( dict_data['idx_loc_in_region'][0].shape[-1]),

                                "d": np.arange( dict_data['input'][0].shape[-1]),
                            }
                            xr_new = xr.Dataset( **kwargs)
                        
                            curr_f_count = len(list(glob.glob( os.path.join(self.cache_path,"*") )))
                            next_cache_fp = os.path.join( self.cache_path, "{:03d}.nc".format(curr_f_count) )
                            xr_new.to_netcdf(next_cache_fp, mode='w', encoding=encoding, unlimited_dims=['sample_idx'] )
                            xr_curr = xr_new
                            gc.collect()
                            
                        elif self.dconfig.data_load_method in ['xarray_single_file_in_mem', 'netcdf4_single_file_on_disk']:
                            
                            kwargs['coords'] = {
                                "sample_idx": np.arange( xr_curr.dims['sample_idx'], xr_curr.dims['sample_idx']+torch.concat(dict_data['input']).shape[0]),

                                "lookback_feat": np.arange( self.dconfig.lookback_feature),
                                "lookback_target": np.arange( self.dconfig.lookback_target),

                                "h_feat": np.arange( dict_data['input'][0].shape[-3]),
                                "w_feat": np.arange( dict_data['input'][0].shape[-2]),

                                "h_target": np.arange( dict_data['target'][0].shape[-2]),
                                "w_target": np.arange( dict_data['target'][0].shape[-1]),
                                "h_w": np.arange( dict_data['idx_loc_in_region'][0].shape[-1]),

                                "d": np.arange( dict_data['input'][0].shape[-1]),
                            }
                            xr_new = xr.Dataset( **kwargs)
                            
                                                          
                            xr_curr = xr.concat( [ xr_curr, xr_new], dim="sample_idx", join="exact" )
                            
                            if self.dconfig.data_load_method == 'netcdf4_single_file_on_disk':
                                
                                if xr_curr.dims['sample_idx'] < 55000: #30000
                                    
                                    continue
                                    
                                # clear xr_curr and cache current data to dataset    
                                else:
                                    for k in kwargs['data_vars'].keys():
                                        nd4_ds.variables[k][ nd4_ds[k].shape[0]:, ...] = copy.deepcopy(xr_curr[k].data)
                                    
                                    # resetting the xr_curr 
                                    xr_curr = xr.Dataset(
                                        coords={
                                            "sample_idx": np.arange( xr_curr.dimensions.sample_idx.size, ) ,
                                            "lookback_feat": np.arange( self.dconfig.lookback_feature),
                                            "lookback_target": np.arange( self.dconfig.lookback_target),
                                            "h_feat": np.arange( dict_data['input'][0].shape[-3]),
                                            "w_feat": np.arange( dict_data['input'][0].shape[-2]),
                                            "d": np.arange( dict_data['input'][0].shape[-1]),
                                            "h_target": np.arange( dict_data['target'][0].shape[-2]),
                                            "w_target": np.arange( dict_data['target'][0].shape[-1]),
                                            "h_w": np.arange( dict_data['idx_loc_in_region'][0].shape[-1] )
                                            },
                                        data_vars={                                
                                                'input': (("sample_idx","lookback_feat","h_feat","w_feat","d"), np.zeros_like( np.concatenate(dict_data['input']))[:0]),
                                                "target": (("sample_idx","lookback_target","h_target","w_target"), np.zeros_like( np.concatenate(dict_data['target']) )[:0] ),
                                                "mask": (("sample_idx","lookback_target","h_target","w_target"), np.zeros_like( np.concatenate(dict_data['mask']))[:0]),
                                                "idx_loc_in_region":(("sample_idx","h_w"), np.zeros_like( np.concatenate(dict_data['idx_loc_in_region']))[:0]),
                                                "li_locations":(("sample_idx","lookback_target"), np.concatenate(dict_data['li_locations']))[:0] ,
                                                "target_date_window": ( ("sample_idx",'lookback_target'),np.zeros_like( np.concatenate(dict_data['target_date_window']) )[:0] )
                                            } )
                                
                                                            
                if bool_update_scaler_features: 
                    # reshaping feature into ( num, dims) dimension required by partial_fit
                    dim = dict_data['input'][0].shape[-1]
                    features = torch.concat(dict_data['input'])
                    features_numpy = features.numpy()
                    # make sure to only take inner dim if applicable and to use feature_mask
                    if features_numpy.ndim == 5:
                        bounds = Era5EobsDataset.central_region_bounds(self.dconfig) #list [ lower_h_bound[0], upper_h_bound[0], lower_w_bound[1], upper_w_bound[1] 
                        features_numpy = torch.stack( [
                            Era5EobsDataset.extract_central_region(feature, bounds )
                                for feature in torch.unbind(features, dim=-1 )
                                ]
                            ,dim=-1)

                    self.scaler_features.partial_fit( features_numpy.reshape(-1, dim ) )
                
                if bool_update_scaler_target:
                    dim = 1    
                    target = torch.concat(dict_data['target'])
                    target_mask = torch.concat(dict_data['mask'])

                    target = target[~target_mask]
                    if target.numel()!=0:
                        self.scaler_target.partial_fit( target.reshape(-1, dim).numpy() )

                sample_idx += torch.concat(dict_data['input']).shape[0]
                
                # yield dict_data
                if type(dict_data['input'])==tuple:
                    li_dicts = [ {key:dict_data[key][idx] for key in dict_data.keys() } for idx in range(len(dict_data['target'])) ]
                    yield from li_dicts
                
                else:
                    yield dict_data
             
            if not self.cache_exists:
                
                if self.dconfig.data_load_method in ['netcdf4_single_file_on_disk']:
                    for k in kwargs['data_vars'].keys():
                        nd4_ds[k][ nd4_ds[k].shape[0]: , ...] = copy.deepcopy(xr_curr[k].data)
                    
                    # Now to set the length of the sample_idx dimension
                    nd4_ds.close()
                    del nd4_ds
                    del kwargs['data_vars']
                    gc.collect()
                    
                elif self.dconfig.data_load_method in ['xarray_single_file_in_mem']:
                    comp = dict(zlib=True, complevel=9)
                    encoding = {var: comp for var in xr_curr.data_vars if (True or var in ['input']) }
                    xr_curr.to_netcdf(self.cache_path, mode='w', encoding=encoding )
                
            # Implement a scheme that saves the cache in chunks to prevent it getting too large for memory
            # - check if you can append a array to one in memory without loding into RAM
                
    def cache_shuffle(self, li_dicts,  target_sub_idx_increment, feature_sub_idx_increment ):
        # shuffling data 
        # unbundling weekly batches and shifting data forward n days 
        # such that the weekly periods are different
        keys = list(li_dicts[0].keys())

        dict_loc_unbundled = {}
        dict_loc_batched = {}

        # Unbundling each variable for each location
        # Converting it into one long sequence for each variable for a location instead of chunks of 7 day data
        locations = self.locations if self.locations != ['All'] else Generator.get_locs_for_whole_map( self.dconfig)
        locations = list(map(str,locations))
        for loc in locations:
            loc_dict = OrderedDict()

            try:
                location_data = [ dict_ for dict_ in li_dicts if all((l == loc for l in dict_["li_locations"])) ]
            except ValueError as e:
                location_data = [ dict_ for dict_ in li_dicts if (dict_["li_locations"]==loc).all() ]
            
            if len( location_data ) == 0: 
                continue

            for key in keys:
                
                #Concat all elements for each variable
                if isinstance( li_dicts[0][key], torch.Tensor ):
                    
                    if key in ["input"]:
                        key_data = torch.cat( [_dict[key].squeeze(0) for _dict in location_data], dim=0 ) 

                    elif key in ["idx_loc_in_region"]:
                        key_data = torch.stack( [_dict[key].squeeze(0) for _dict in location_data], dim=0 )
                    
                    elif key in ['target']:
                        key_data = torch.cat( [_dict[key].squeeze(0) for _dict in location_data], dim=0 ) 
                    
                    elif key in ['mask']:
                        key_data = torch.cat( [_dict[key].squeeze(0) for _dict in location_data], dim=0 )
                                            
                elif isinstance( li_dicts[0][key], list ):                    
                    key_data = sum( [_dict[key] for _dict in location_data], [] )
                                
                loc_dict[key] = key_data
                    
            dict_loc_unbundled[loc] = loc_dict
        
        # rebatching into weeks with a n day increment
        for loc in dict_loc_unbundled.keys():
            dict_loc_batched[loc] = OrderedDict()
            # incrementing and batching
            for key in keys:
                
                if key in ['target','mask']:
                    d = dict_loc_unbundled[loc][key] = dict_loc_unbundled[loc][key][target_sub_idx_increment:]
                    l = len(d)-self.dconfig.lookback_target
                    dict_loc_batched[loc][key] = [ d[idx:idx+self.dconfig.lookback_target].unsqueeze(0) for idx in range(0, l, self.dconfig.lookback_target) ]
                
                elif key in ['li_locations','target_date_window']:
                    d = dict_loc_unbundled[loc][key] = dict_loc_unbundled[loc][key][target_sub_idx_increment:]
                    l = len(d)-self.dconfig.lookback_target
                    dict_loc_batched[loc][key] = [ d[idx:idx+self.dconfig.lookback_target] for idx in range(0, l, self.dconfig.lookback_target) ]

                elif key in ['idx_loc_in_region']:
                    # idx_loc_in_region only has one value per lookback_target.
                    s_dix = round( target_sub_idx_increment/self.dconfig.lookback_target )
                    d = dict_loc_unbundled[loc][key][s_dix:]
                    l = len(d)
                    dict_loc_batched[loc][key] = [ d[idx:idx+1] for idx in range(0, l, 1) ]

                elif key in ['input']:
                    dict_loc_unbundled[loc][key] = dict_loc_unbundled[loc][key][feature_sub_idx_increment:]
                    d = dict_loc_unbundled[loc][key]
                    l = len(d)-self.dconfig.lookback_feature
                    dict_loc_batched[loc][key] = [ d[idx:idx+self.dconfig.lookback_feature].unsqueeze(0) for idx in range(0, l, self.dconfig.lookback_feature) ]

        count = sum( len( dict_loc_batched[loc][keys[0]] ) for loc in dict_loc_batched.keys() )
        li_dicts_shuffled = [ {} for i in range(count) ]
        
        # Creating a list of dicts structure - where each dict contains data for one 7 day period
        for key in keys:
            li_datums = sum( [ dict_loc_batched[loc][key] for loc in dict_loc_batched.keys() ], [] )

            for idx in range(count):
                li_dicts_shuffled[idx][key] = li_datums[idx]
            # for idx, shuffled_idx in enumerate():
            #     li_dicts_shuffled[idx][key] = li_datums[shuffled_idx]
        random.shuffle(li_dicts_shuffled) 
        
        return li_dicts_shuffled

    def preprocess_batch(self, feature, feature_mask, target, target_mask, target_date_window, normalize=True):
        
        # Converting to tensors
        feature = torch.as_tensor(feature)
        feature_mask = torch.as_tensor(feature_mask)
        target = torch.as_tensor(target)
        target_mask = torch.as_tensor(target_mask)

        # Preparing feature model fields
        # unbatch
        feature = feature.view(-1, self.dconfig.lookback_feature, *feature.shape[-3:] ) # ( bs/feat_lookback, feat_days ,h, w, shape_dim)
        feature_mask = feature_mask.view(-1, self.dconfig.lookback_feature,*feature.shape[-3:] ) # ( bs/feat_lookback ,h, w, shape_dim)
        
        if normalize:
            feature = (feature-self.feature_mean )/self.features_scale
        feature.masked_fill_( feature_mask, self.dconfig.mask_fill_value['model_field'])
        # feature.masked_fill_( target_mask, self.dconfig.mask_fill_value['model_field'])

        # Preparing Eobs and target_rain_data
        target = target.view(-1, self.dconfig.lookback_target, *target.shape[-2:] ) #( bs, target_periods ,h1, w1, target_dim)
        target_mask = target_mask.view(-1, self.dconfig.lookback_target, *target.shape[-2:] )#( bs*target_periods ,h1, w1, target_dim)
        target_date_window = target_date_window.reshape( (-1,self.dconfig.lookback_target  ) )
        
        if normalize:
            target = target*self.target_scale
        target.masked_fill_(target_mask, self.dconfig.mask_fill_value['rain'] )

        li_feature, li_target, li_target_mask, idx_loc_in_region, li_locs, li_target_date_window = self.location_extractor( feature, target, target_mask, self.locations, target_date_window )
        
        dict_data = { k:v for k,v in zip(['input','target','mask','idx_loc_in_region','li_locations','target_date_window'], [li_feature, li_target, li_target_mask, idx_loc_in_region, li_locs, li_target_date_window] ) }
        return dict_data

    def get_idx(self, date:Union[np.datetime64,str]):
        """ Returns two indexes
                The first index is the idx at which to start extracting data from the feature dataset
                The second index is the idx at which to start extracting data from the target dataset
            Args:
                start_date (np.datetime64): Start date for evaluation
            Returns:
                tuple (int, int): starting index for the feature, starting index for the target data
        """        

        if type(date)==str:
            date = np.datetime64(date)
            
        feature_start_date = self.dconfig.feature_start_date
        target_start_date = self.dconfig.target_start_date

        feat_days_diff = np.timedelta64(date - feature_start_date,'6h').astype(int)
        tar_days_diff = np.timedelta64(date - target_start_date, 'D').astype(int)

        feat_idx = feat_days_diff #since the feature comes in four hour chunks
        tar_idx = tar_days_diff 

        return feat_idx, tar_idx

    def location_extractor(self, feature, target, target_mask, locations, target_date_window ):
        """Extracts the temporal slice of patches corresponding to the locations of interest 

                Args:
                    ds (tf.Data.dataset): dataset containing temporal slices of the regions surrounding the locations of interest
                    locations (list): list of locations (strings) to extract

                Returns:
                    tuple: (tf.data.Dataset, [int, int] ) tuple containing dataset and [h,w] of indexes of the central region
        """        
                
        # list of central h,w indexes from which to extract the region around
        if locations == ["All"]:
            li_hw_idxs = Generator.get_locs_for_whole_map( self.dconfig ) #[ ([upper_h, lower_h]. [left_w, right_w]), ... ]
            
            locs_latlon_for_whole_map = Generator.get_locs_latlon_for_whole_map(self.dconfig) #[ ([lat1, lat2]. [lon1, lon2]), ... ]

            #Convert to string format            
            li_hw_idxs_str = [ str(hw_idxs) for hw_idxs in li_hw_idxs ]
            
            locations_ = copy.deepcopy(li_hw_idxs_str)
        
        elif locations[0] in Generator.get_locs_for_whole_map(self.dconfig):
            li_hw_idxs = copy.deepcopy(locations)
            #Convert to string format            
            li_hw_idxs_str = [ str(hw_idxs) for hw_idxs in li_hw_idxs ]
            
            locations_ = li_hw_idxs_str
        
        elif isinstance( locations[0], string_classes ):
            li_hw_idxs = [ self.rain_data.find_idx_of_loc_region( _loc, self.dconfig ) for _loc in locations ] #[ (h_idx,w_idx), ... ]
            #li_locs = np.repeat(locations, len(target) )
            locations_ = copy.deepcopy(locations)           
        
        # Creating seperate datasets for each location
        li_feature, li_target, li_target_mask = zip(*[self.select_region(feature, target, target_mask, hw_idxs[0], hw_idxs[1]) for hw_idxs in li_hw_idxs ] )
                

        
        lcs = locations_ 
        lit = li_target
        li_locs = [
                    [
                        lcs[i:i+1]*lit[i].shape[1]
                    ]*lit[i].shape[0] 
                    for i in range(len(lit))
                   ]
                # li_target_date_window= np.tile( target_date_window[np.newaxis, ...], ( len(lit), 1, 1) )
        li_target_date_window = [target_date_window]*len(lit)
        
        # pair of indexes locating the central location within the grid region extracted for any location
        idx_loc_in_region = [ np.floor_divide( self.dconfig.outer_box_dims, 2)[np.newaxis,...] ]*len( torch.concat(li_feature) ) #This specifies the index of the central location of interest within the (h,w) patch    
        
        return li_feature, li_target, li_target_mask, idx_loc_in_region, li_locs, li_target_date_window
    
    def select_region( self, mf, rain, rain_mask, h_idxs, w_idxs):
        """ Extract the region relating to a [h_idxs, w_idxs] pair

            Args:
                mf:model field data
                rain:target rain data
                rain_mask:target rain mask
                h_idxs:int
                w_idxs:int

            Returns:
                tf.data.Dataset: 
        """

        """
            idx_h,idx_w: refer to the top left right index for the square region of interest this includes the region which is removed after cropping to calculate the loss during train step
        """
    
        mf = mf[ ..., h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ,:] # (shape, h, w, d)
        rain = rain[ ..., h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
        rain_mask = rain_mask[ ..., h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
            
        return mf, rain, rain_mask #Note: expand_dim for unbatch/batch compatibility

    def create_cache_scaler(self):

        """
            This functions returns the cache_path, scaler_features, scaler_target
            if self.scaler_features, or self.scaler_target do not exists then they are created

        Returns:
            [type]: [description]
        """
        
        # if os.path.exists(self.cache_path):
        #     os.remove(self.cache_path)
        # self.cache_exists = False

        for dict_data in iter(self):
            pass

        self.cache_exists = True
        
        assert self.scaler_features 
        assert self.scaler_target
        assert os.path.exists(self.cache_path) 

        return self.scaler_features, self.scaler_target

    def create_cache_params(self):
        
        if os.path.isfile(self.cache_path):
            with xr.open_dataset( self.cache_path, cache=True ) as xr_cache:
                # for feature, target, target_mask, idx_loc_in_region in self.cached_data:
                if not hasattr(self,'cache_len'):
                    self.cache_len = xr_cache.dims['sample_idx']
        
        elif os.path.isdir(self.cache_path):
            # aggregating lengths of all files
            if not hasattr(self,'max_cache_len'):
                
                with xr.open_mfdataset( sorted(list( glob.glob(self.cache_path+"/*"))), 
                                      decode_cf=self.xarray_decode, 
                                      decode_times=self.xarray_decode,  
                                        concat_dim='sample_idx',    
                                        combine='nested',     
                                        #coords=['sample_idx'],                             
                                        cache=True ) as xr_cache:
                    self.cache_len = xr_cache.dims['sample_idx']
                    
        gc.collect()
        
        # Making sure its a multiple of lookback away from start idx
        if not hasattr(self,'cache_start_idx'):
            self.cache_start_idx = 0
        if not hasattr(self,'cache_end_idx'):
            # if self.shuffle:
            #     self.cache_end_idx = 0 + int(((self.max_cache_len - self.cache_start_idx)//self.dconfig.lookback_target)*self.dconfig.lookback_target )
            # else:
            self.cache_end_idx = 0 + self.cache_len
        # if not hasattr(self,'cache_len'):
        #     self.cache_len = self.cache_end_idx + 1 - self.cache_start_idx

    @staticmethod
    def worker_init_fn(worker_id:int):
        
        worker_info = torch.utils.data.get_worker_info()
        worker_count = worker_info.num_workers

        if isinstance(worker_info.dataset, ShufflerIterDataPipe):
            # offers backward compatability for pytorch lightning pre v
            ds = worker_info.dataset.datapipe.iterable
        elif isinstance(worker_info.dataset, Era5EobsDataset ):
            ds = worker_info.dataset
        elif isinstance(worker_info.dataset, _IterDataPipeSerializationWrapper ):
            ds = worker_info.dataset._datapipe.datapipe.iterable
        else:
            raise ValueError

        if ds.cache_exists:
            per_worker = ds.cache_len // worker_count
            ds.cache_start_idx =  per_worker * worker_id
            ds.cache_end_idx = per_worker * ( worker_id + 1 ) if (worker_id+1!=worker_count) else ds.cache_len
            ds.cache_len_per_worker = per_worker
            
        else:
            # Changing the start_idx and end_idx in the underlying generators
            mf_data_len_per_location = ds.mf_data.start_idx - ds.mf_data.end_idx
            per_worker_per_location = mf_data_len_per_location//worker_count
            ds.mf_data.start_idx = worker_id*per_worker_per_location
            ds.mf_data.end_idx = (worker_id+1)*per_worker_per_location
            ds.mf_data.data_len_per_worker = per_worker_per_location * ds.loc_count

            rain_data_len_per_location = ds.rain_data.start_idx - ds.rain_data.end_idx
            per_worker_per_location = rain_data_len_per_location//worker_count 
            ds.rain_data.start_idx = worker_id*per_worker_per_location
            ds.rain_data.end_idx = (worker_id+1)*per_worker_per_location
            ds.rain_data.data_len_per_worker = per_worker_per_location * ds.loc_count
        
    @staticmethod
    def parse_data_args(parent_parser=None,list_args=None ):
        
        if parent_parser != None:
            parser = argparse.ArgumentParser(
                parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)
        else:
            parser = argparse.ArgumentParser( add_help=True, allow_abbrev=False )

        parser.add_argument("--original_uk_dim", default=(100,140) )
        parser.add_argument("--input_shape", default=(6,), type=tuple_type ) #TODO: need to roll together input_shape and outer_box_dim logic into one variable. Currently it only refers to the channel depth variable 
        parser.add_argument("--output_shape", default=(1,1), type=tuple_type ) #NOTE: this refers to the depth of the output i.e. do we predict mean and var term or just mean

        parser.add_argument("--locations", nargs='+', required=False, default=[])
        parser.add_argument("--locations_test", nargs='+', required=False, 
                                default=[] )

        parser.add_argument("--data_dir", default="./Data/uk_rain", type=str)
        parser.add_argument("--rain_fn", default="eobs_true_rainfall_197901-201907_uk.nc", type=str)
        parser.add_argument("--mf_fn", default="model_fields_linearly_interpolated_1979-2019.nc", type=str)

        parser.add_argument("--vertical_shift", type=int, default=4)
        parser.add_argument("--horizontal_shift", type=int, default=4)

        parser.add_argument("--outer_box_dims", default=[16,16], type=tuple_type)
        parser.add_argument("--inner_box_dims", default=[4,4], type=tuple_type)

        parser.add_argument("--lookback_target", type=int, default=7)
        # parser.add_argument("--target_range", nargs='+', default=[0,4])

        parser.add_argument("--train_start", type=str, default="1979")
        parser.add_argument("--train_end", type=str, default="1995-03")

        parser.add_argument("--val_start", type=str, default="1995-03")
        parser.add_argument("--val_end", type=str, default="1999-07")

        parser.add_argument("--test_start", type=str, default="1999")
        parser.add_argument("--test_end", type=str, default="2019-07")

        parser.add_argument("--min_rain_value", type=float, default=0.5)
        parser.add_argument("--gen_size", type=int, default=50, help="Chunk size when slicing the netcdf fies for model fields and rain. When training over many locations, make sure to use large chunk size.")
        parser.add_argument("--gen_size_test", type=int, default=50, help="Chunk size when slicing the netcdf fies for model fields and rain. When training over many locations, make sure to use large chunk size.")
        parser.add_argument("--cache_gen_size", type=int, default=600, help="Chunk size when slicing the netcdf fies for model fields and rain. When training over many locations, make sure to use large chunk size.")
        parser.add_argument("--cache_gen_size_test", type=int, default=600, help="Chunk size when slicing the netcdf fies for model fields and rain. When training over many locations, make sure to use large chunk size.")
        
        parser.add_argument("--shuffle", type=lambda x: bool(int(x)), default=True, choices=[0,1] )
        
        # parser.add_argument("--memory_effecient", type=lambda x: bool(int(x)), default=False, choices=[0,1] )
        parser.add_argument("--data_load_method", type=str, default='xarray_mult_files_on_disk', 
                            choices=['xarray_mult_files_on_disk', 'xarray_single_file_in_mem', 'netcdf4_single_file_on_disk'])
        
        dconfig = parser.parse_known_args(args=list_args)[0]
        
        dconfig.locations = sorted(dconfig.locations)
        dconfig.locations_test = sorted(dconfig.locations_test)
        
        #TODO: Add check to ensre that original_shape dims is divisible by  inner_box dims and outer_box dims
        ## iff using two dimensions in input shape or implement method whereby overall shape is cropped to allows for any input_shape dims
        
        dconfig = Era5EobsDataset.add_fixed_args(dconfig)
        
        return dconfig
    
    @staticmethod
    def add_fixed_args(dconfig):
        dconfig.mask_fill_value = {
                                    "rain":0.0,
                                    "model_field":0.0 
        }
        dconfig.vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
        
        dconfig.window_shift = dconfig.lookback_target
        
        #Input is at 6 hour intervals, target is daily
        dconfig.lookback_feature = dconfig.lookback_target*4
        
        dconfig.target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
        dconfig.feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')

        # a string containing four dates seperated by underscores
        # The numbers correspond to trainstart_trainend_valstart_valend

        train_start_date = np.datetime64(dconfig.train_start,'D')
        train_end_date = (pd.Timestamp(dconfig.train_end) - pd.DateOffset(seconds=1) ).to_numpy()

        val_start_date = np.datetime64(dconfig.val_start,'D')
        val_end_date = (pd.Timestamp(dconfig.val_end) - pd.DateOffset(seconds=1) ).to_numpy()

        test_start_date = np.datetime64(dconfig.test_start,'D')
        test_end_date = (pd.Timestamp(dconfig.test_end) - pd.DateOffset(seconds=1) ).to_numpy()

        if len(dconfig.locations)== 0:
            pass
        
        elif dconfig.locations[0][:13] == "WholeMapSplit":
            all_hw_idxs = Generator.get_locs_for_whole_map(dconfig)
            
            train_prop, test_prop = dconfig.locations[0][14:].split('_')
            train_prop = float(train_prop)/100
            test_prop = float(test_prop)/100
            
            train_count = int( len(all_hw_idxs) * train_prop )
            test_count = int( len(all_hw_idxs) * test_prop )

            #For reproducibility
            random.seed(24)
            train_test_group = random.sample( all_hw_idxs , train_count + test_count ) 
            random.seed(24)
            random.shuffle(train_test_group)
                        
            train_group, test_group = train_test_group[:train_count], train_test_group[ train_count: ]
            
            dconfig.locations = train_group
            dconfig.locations_test = test_group
            
        elif dconfig.locations[0] == "All_Cities":
            dconfig.locations = sorted( list( Generator.city_latlon.keys() ) )
          
        if len(dconfig.locations_test)==0:
            pass
                                     
        elif dconfig.locations_test[0][:13] == "WholeMapSplit":
            all_hw_idxs = Generator.get_locs_for_whole_map(dconfig)
            
            train_prop, test_prop = dconfig.locations[0][14:].split('_')
            train_prop = float(train_prop)/100
            test_prop = float(test_prop)/100
            
            train_count = int( len(all_hw_idxs) * train_prop )
            test_count = int( len(all_hw_idxs) * test_prop )

            #For reproducibility
            random.seed(24)
            train_test_group = random.sample( all_hw_idxs , train_count + test_count ) 
            random.seed(24)
            random.shuffle(train_test_group)
                        
            train_group, test_group = train_test_group[:train_count], train_test_group[ train_count: ]
            
        elif dconfig.locations_test[0] == "All_Cities":
            dconfig.locations_test = sorted( list( Generator.city_latlon.keys() ) )


        loc_count = len(dconfig.locations)  if \
                    dconfig.locations != ["All"] else \
                    len( Generator.get_locs_for_whole_map(dconfig) )

        loc_count_test = loc_count \
                            if not dconfig.locations_test \
                            else (
                                    len(dconfig.locations_test)  if \
                                        dconfig.locations_test != ["All"] else \
                                        len( Generator.get_locs_for_whole_map(dconfig))
                                    )
        
        dconfig.loc_count = loc_count
        dconfig.loc_count_test = loc_count_test

        dconfig.train_set_size_elements = ( np.timedelta64(train_end_date - train_start_date,'D') // dconfig.window_shift ).astype(int)
        dconfig.train_set_size_elements *= loc_count

        dconfig.val_set_size_elements = ( np.timedelta64(val_end_date - val_start_date,'D')  // dconfig.window_shift  ).astype(int)               
        dconfig.val_set_size_elements *= loc_count

        dconfig.test_set_size_elements = ( np.timedelta64(test_end_date - test_start_date,'D')  // dconfig.window_shift  ).astype(int)               
        dconfig.test_set_size_elements *= loc_count_test
               

        return dconfig

    @staticmethod
    def cond_rain(vals, probs, threshold=0.5):
        """
            If prob of event occuring is above 0.5 return predicted conditional event value,
            If it is below 0.5, then return 0
        """
        round_probs = torch.where(probs<=threshold, 0.0, 1.0)
        vals = vals* round_probs
        return vals
    
    @staticmethod
    def central_region_bounds(dconfig):
        """Returns the indexes defining the boundaries for the central regions for evaluation

        Args:
            dconfig (dict): information on formualation of the patches used in this ds 

        Returns:
            list: defines the vertices of the patch for extraction
        """    

        central_hw_point = np.asarray(dconfig.outer_box_dims)//2
        
        lower_hw_bound = central_hw_point - np.asarray(dconfig.inner_box_dims) //2

        upper_hw_bound = lower_hw_bound + np.asarray(dconfig.inner_box_dims )
        

        return [lower_hw_bound[0], upper_hw_bound[0], lower_hw_bound[1], upper_hw_bound[1]]

    @staticmethod
    def extract_central_region(tensor, bounds):
        """
            Args:
                tensor ([type]): 4d 
                bounds ([type]): bounds defining the vertices of the patch to be extracted for evaluation
        """
        tensor_cropped = tensor[ ..., bounds[0]:bounds[1],bounds[2]:bounds[3]  ]     #(bs, h , w)
        return tensor_cropped
    
    @staticmethod
    def water_mask( tensor, mask, mask_val=np.nan):
        """Mask out values in tensor by with mask value=0.0
        """
        tensor = torch.where(mask, tensor, mask_val)

        return tensor
# endregion

def scale(x):
    return np.log10(x+1)
def inv_scale(x):
    return  np.power(10,x)-1

class Era5EobsTopoDataset_v2(Dataset):
    """
        This version extends Era5EobsDataset in the following ways:
            - The Generator_mf returns the un-upscaled 20x24 model fields map of the UK
            - Also Attachs the topology of the UK as an output
            - The output of the generator includes the un-upscaled 20x24 modelfields
            
        Similar to before
            - outputs Era5 rain

    """
    latitude_array = np.linspace(58.95, 49.05, 100)
    longitude_array = np.linspace(-10.95, 2.95, 140)
    
    def __init__(self, start_date, end_date,  
                    dconfig,
                    xarray_decode=False,
                    scaler_features=None,
                    scaler_target=None,
                    scaler_topo=None
                    ) -> None:
        super().__init__()

        #'/mnt/Data1/akann1w0w1ck/NeuralGLM/Data
        self.path_to_rain = os.path.join( dconfig.data_dir,'uk_rain/eobs_true_rainfall_197901-201907_uk.nc')
        self.path_to_elevation = os.path.join( dconfig.data_dir,'uk_rain/topo_0.1_degree.grib')
        self.path_to_fields = os.path.join( dconfig.data_dir,'uk_rain/model_fields_1979-2019.nc')
        self.dconfig = dconfig
        self.xarray_decode = xarray_decode
        
        start_date_idx_mf, start_date_idx_rain = self.get_idx(start_date)
        end_date_idx_mf, end_date_idx_rain = self.get_idx(end_date)
        
        self.max_data_len_rain = xr.open_dataset(self.path_to_rain, decode_times=False, decode_cf=False)['time'].size
        self.start_idx_rain = start_date_idx_rain 
        self.end_idx_rain = end_date_idx_rain if end_date_idx_rain else self.start_idx_rain + self.max_data_len_rain
    
        self.max_data_len_mf = xr.open_dataset(self.path_to_fields, decode_times=False, decode_cf=False)['time'].size
        self.start_idx_mf = start_date_idx_mf
        self.end_idx_mf = end_date_idx_mf if end_date_idx_mf else self.start_idx_mf + self.max_data_len_mf
        
        self.rain_dset_time_freq = 'D'
        self.rain_date_range = np.asarray( pd.date_range( start= start_date, periods=self.max_data_len_rain, freq=self.rain_dset_time_freq, normalize=True  ) )

        self.mf_dset_time_freq = '6H'
        self.mf_date_range = np.asarray( pd.date_range( start= start_date, periods=self.max_data_len_mf, freq=self.mf_dset_time_freq, normalize=True  ) )
        self.vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]       
        
        self.scaler_features=scaler_features
        self.scaler_target=scaler_target
        self.scaler_topo=scaler_topo
        
        self.target_norm_method = dconfig.target_norm_method
        
        self.load_dataset_rain()
        self.load_dataset_mf()
        self.load_dataset_topo()
    
    @lru_cache
    def __len__(self, ):
        return len(self.rain_data)
        
    def __getitem__(self, idx):
                    
        # Input
        feature = torch.tensor(self.mf_data[idx*4 : idx*4+4])
        feature_mask = torch.tensor(self.mf_mask[idx*4 : idx*4+4])
        
        feature = (feature-self.features_mean)/(self.features_scale)
        feature.masked_fill_( feature_mask, self.dconfig.mask_fill_value['model_field'])
        
        topo = torch.tensor(self.topo_data[None,... ])
        topo = topo * self.scaler_topo_scale_
        
        # Target
        target = torch.tensor(self.rain_data[idx:idx+1])
        target_mask = torch.tensor(self.rain_mask[idx:idx+1])
        target_windows = self.rain_date_windows[idx:idx+1]
        
        target.masked_fill_(target_mask, self.dconfig.mask_fill_value['rain'] )
        
        if self.target_norm_method == 'log':
            target = torch.log10(target+1)
        elif self.target_norm_method == 'scale':
            target = (target * self.scaler_target_scale)
            pass
        
        

        # Score        
        dict_data = { 'fields':feature, 'rain':target, 'topo':topo,
                        'mask':target_mask, 'target_date_window':target_windows
                        }

        return dict_data #'input','target','mask','idx_loc_in_region','li_locations','target_date_window'
    
    def load_dataset_rain(self):
        slice_t = slice( self.start_idx_rain, self.end_idx_rain )
        slice_h = slice( None , None, -1 )
        slice_w = slice( None, None )
        date_windows  = np.asarray( self.rain_date_range[slice_t] )
        
        with xr.open_dataset( self.path_to_rain, decode_times=self.xarray_decode, decode_cf=self.xarray_decode ) as xr_gn:
            marray = xr_gn.isel(time=slice_t ,latitude=slice_h, longitude=slice_w)['rr'].to_masked_array(copy=True)
            array, mask = np.ma.getdata(marray), np.ma.getmask(marray)
            mask = (array==9.969209968386869e+36)
        
            self.rain_data = array
            self.rain_mask = mask
            self.rain_date_windows = date_windows
        
        if self.target_norm_method == 'scale':
            if getattr(self, 'scaler_target', None) is None:
                self.scaler_target = MinMaxScaler(feature_range=(0.0, 90.0))
                self.scaler_target.fit( array[~mask].reshape(-1,1) )
            self.scaler_target_scale = torch.as_tensor( self.scaler_target.scale_[0] )
            
                    
        if self.target_norm_method == 'log':
            if getattr(self, 'scaler_target', None) is None:
                self.scaler_target = FunctionTransformer(func= scale , inverse_func= inv_scale )
                self.scaler_target.fit( array[~mask].reshape(-1,1) )
            self.target_scale = lambda x: torch.log10(x+1)
                
    def load_dataset_mf(self):
        slice_t = slice( self.start_idx_mf, self.end_idx_mf )
        slice_h = slice( None , None )
        slice_w = slice( None, None )
        date_windows  = np.asarray( self.mf_date_range[slice_t] )
        
        with xr.open_dataset( self.path_to_fields, decode_times=False, decode_cf=False ) as xr_gn:
            
            marray = [ xr_gn[name].isel(time=slice_t ,latitude=slice_h, longitude=slice_w).to_masked_array(copy=True) for name in self.vars_for_feature  ]
            list_datamask = [(np.ma.getdata(_mar), np.ma.getmask(_mar)) for _mar in marray]
            _data, _masks = list(zip(*list_datamask))
            
            stacked_data = np.stack(_data, axis=-1)
            stacked_masks = np.stack(_masks, axis=-1)
                        
            self.mf_data = stacked_data[ :, :, :, :]
            self.mf_mask = stacked_masks[ :, :, :, :]

        if getattr(self, 'scaler_features', None) is None:
            self.scaler_features = StandardScaler()
            self.scaler_features.fit(  self.mf_data.reshape(-1, len(self.vars_for_feature) )  )
        
        self.features_scale = torch.as_tensor( self.scaler_features.scale_ )
        self.features_mean = torch.as_tensor( self.scaler_features.mean_ )
    
    def load_dataset_topo(self):
                
        with xr.open_dataset( self.path_to_elevation, engine='cfgrib', decode_times=False, decode_cf=False ) as xr_gn:            
            marray = xr_gn['z'].to_masked_array(copy=True)
            array,_ = np.ma.getdata(marray), np.ma.getmask(marray)
                                    
            self.topo_data = array[ 2:-2, 2:-2]
        
        if getattr(self, 'scaler_topo', None) is None:
            self.scaler_topo = MinMaxScaler(feature_range=(0,0.2))
            self.scaler_topo.fit(self.topo_data.reshape(-1, 1))
            
        self.scaler_topo_scale_ = torch.as_tensor(self.scaler_topo.scale_)
    
    def get_idx(self, date:Union[np.datetime64,str]):
        """ Returns two indexes
                The first index is the idx at which to start extracting data from the feature dataset
                The second index is the idx at which to start extracting data from the target dataset
            Args:
                start_date (np.datetime64): Start date for evaluation
            Returns:
                tuple (int, int): starting index for the feature, starting index for the target data
        """        

        if type(date)==str:
            date = np.datetime64(date)
            
        feature_start_date = self.dconfig.feature_start_date
        target_start_date = self.dconfig.target_start_date

        feat_diff = np.timedelta64(date - feature_start_date,'6h').astype(int)
        tar_diff = np.timedelta64(date - target_start_date, 'D').astype(int)

        feat_idx = feat_diff #since the feature comes in four hour chunks
        tar_idx = tar_diff 

        return feat_idx, tar_idx    
    

    @staticmethod
    def get_datasets( dconfig, **kwargs):

        ds_train = Era5EobsTopoDataset_v2( start_date = dconfig.train_start, 
                                    end_date=dconfig.train_end,
                                    dconfig=dconfig
                                    )
        
        ds_val = Era5EobsTopoDataset_v2( start_date=dconfig.val_start,
                                        dconfig=dconfig,
                                        end_date=dconfig.val_end,
                                        
                                        scaler_features= ds_train.scaler_features,
                                        scaler_target=ds_train.scaler_target,
                                        scaler_topo=ds_train.scaler_topo
                                        )
        
        
        ds_test = Era5EobsTopoDataset_v2( start_date=dconfig.test_start, 
                                         end_date=dconfig.test_end,
                                        dconfig=dconfig,
                                        
                                        scaler_features= ds_train.scaler_features,
                                        scaler_target=ds_train.scaler_target,
                                        scaler_topo=ds_train.scaler_topo,
                                        xarray_decode=True
                                        )

        return ds_train, ds_val, ds_test, ds_train.scaler_features, ds_train.scaler_target, ds_train.scaler_topo
    
    @staticmethod
    def collate_fn( batch ):
        
        fields = torch.concat([ d['fields'] for d in batch ], 0)
        rain = torch.concat([ d['rain'] for d in batch ], 0)
        topo = torch.concat([ d['topo'] for d in batch ], 0)
        mask = torch.concat([ d['mask'] for d in batch ], 0)
        target_date_window = np.concatenate([ d['target_date_window'] for d in batch ], 0)
        
        dict_data = { 'fields':fields, 'rain':rain, 'topo':topo,             
                     'mask':mask, 'target_date_window':target_date_window }
        
        return dict_data
        
    @staticmethod
    def parse_data_args(parent_parser=None, list_args=None ):
        
        if parent_parser != None:
            parser = argparse.ArgumentParser(
                parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)
        else:
            parser = argparse.ArgumentParser( add_help=True, allow_abbrev=False )

        # parser.add_argument("--original_uk_dim", default=(100,140) )
        parser.add_argument("--input_shape", default=(6,), type=tuple_type ) #TODO: need to roll together input_shape and outer_box_dim logic into one variable. Currently it only refers to the channel depth variable 
        parser.add_argument("--output_shape", default=(1,1), type=tuple_type ) #NOTE: this refers to the depth of the output i.e. do we predict mean and var term or just mean

        parser.add_argument("--data_dir", default="./Data/uk_rain", type=str)
        

        parser.add_argument("--train_start", type=str, default="1979")
        parser.add_argument("--train_end", type=str, default="1993-07")

        parser.add_argument("--val_start", type=str, default="1993-07")
        parser.add_argument("--val_end", type=str, default="1999")

        parser.add_argument("--test_start", type=str, default="1999")
        parser.add_argument("--test_end", type=str, default="2009")
        parser.add_argument("--gen_size", type=int, default=8, help="Chunk size when slicing the netcdf fies for model fields and rain. When training over many locations, make sure to use large chunk size.")

        parser.add_argument("--target_norm_method", type=str, default='log', choices=['log','scale'])
        dconfig = parser.parse_known_args()[0]
        
        dconfig = Era5EobsTopoDataset_v2.add_fixed_args(dconfig)
        
        return dconfig
    
    @staticmethod
    def add_fixed_args(dconfig):
        dconfig.mask_fill_value = {"rain":0.0,
                                    "model_field":0.0 }
        
        dconfig.vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
                
        dconfig.target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
        dconfig.feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
        
        return dconfig
    
MAP_NAME_DSET = {
    # "toy":None,
    # "australia_rain":None,
    "uk_rain":Era5EobsDataset
}