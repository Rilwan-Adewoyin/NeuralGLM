import argparse
import numpy as np
from scipy.interpolate import interp2d, griddata
from dataloaders import Era5EobsDataset, Generator, Generator_mf
from torch.utils.data import DataLoader
from typing import List, Tuple
import collections
import glm_utils

class Interpolate3d():
    
    def __init__(self, x:np.array, y:np.array, z:np.array):
        
        self.li_interp2d = [  interp2d(x, y, feature_map,kind='linear' ) for feature_map in np.split(z, z.shape[-1] )  ]
        
    def __call__( self, x, y ):
        
        li_output = [ interpolate2d(x,y) for interpolate2d in self.li_interp2d ]  #List[# outp 2D array with shape (len(y), len(x))]
        
        output = np.stack(li_output, axis=-1)
        
        return output 
    
class InterpolationMixin():
    # Make a method which makes a prediction given an abritrary long,lat value 
    # - map needs to be interpolated (on the fly), 
    # - take the 16 points surrounding central point retained,
    # - use  trained CompoundPoisson (my version), compound Poisson previous literature, Gamma model to predict for that central region
    
    lat_increment = 0.05
    lon_increment = 0.05
    
    def generate_predictions_at_arbitrary_locs(self, li_latlon, start_date, end_date, workers=2, ckpt_path='/Data/uk_rain/model_fields_linearly_interpolated_1979-2019.nc') -> np.array:
                
        # load the 100,140,6 grid of features between start_date and end_date
        fp=ckpt_path
        all_at_once=True 
        iter_chunk_size=28*10
        lookback=28
        start_idx_feat, _ = self.get_index(start_date)
        end_idx_feat, _ = self.get_index(end_date)
                                        
        gen_mf = Generator_mf( fp=ckpt_path
                                all_at_once=True 
                                iter_chunk_size=28*10
                                lookback=28
                                start_idx =start_idx_feat,
                                end_idx = end_idx_feat
                                 )
        data = gen_mf.yield_all()
                
        intrp_grid_size = getattr(self.dconfig,'outerboxdims',[16,16] )
        
        li_latlon_grids = [ self.get_grid_centred_on_latlon(latlon) for latlon in li_latlon ]
        
        
        li_time = []
        li_interp_vals = []
        li_latlon = []
        
        for timestep in date_range_6hrly:
            
            idx_timestep = 
            
            # Do this for every weather feature - make into a class that holds 6 scaling interpolate.interp2d calsses
            z = feature_data[idx_timestep] #( h, w, d )
            
            interpolater = Interpolate3d(Generator.latitude_array,
                                         Generator.longitude_array,
                                         z)

            # Interpolated grid at one 6hour period
            li_interp_vals_6hr = [ interpolater( latlon_grids ) for latlon_grids in li_latlon_grids ]
            
            #
            
            li_timesteps.append(timestep)
            li_interp_vals.append( li_interp_vals_6hr )   
            

        
        return li_timesteps, li_interp_vals, li_latlon_grids
    
    def get_grid_centered_on_latlon( self, latlon:List[float], intrp_grid_size:List[int]) -> Tuple[np.array, np.array]:
        """return x cordinates and y cordinates for grid
            
        Args:
            latlon (List[float]): [description]

        Returns:
            np.array: [description]
        """
        
        lat_s = latlon[0] - (intrp_grid_size[0]/2)*self.lat_increment
        lon_s = latlon[1] - (intrp_grid_size[1]/2)*self.lat_increment

        lat_e = latlon[0] + (intrp_grid_size[0]/2)*self.lat_increment
        lon_e = latlon[1] + (intrp_grid_size[1]/2)*self.lat_increment
        
        lat = np.linspace( lat_s, lat_e, num=intrp_grid_size[0] )
        lon = np.linspace( lon_s, lon_e, num=intrp_grid_size[1] )
        
        return lat, lon