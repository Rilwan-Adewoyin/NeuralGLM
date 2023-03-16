from dataloaders import Era5EobsDataset
from typing import Union
import torch
# from torch._C import Value
# from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
import pickle
from collections import defaultdict
import os
from better_lstm import LSTM
# import einops
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pathlib import Path
import yaml
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import Dict, Any
import json
import pandas as pd
import numpy as np
# import torch_optimizer as optim
from transformers import get_constant_schedule_with_warmup
# from collections import OrderedDict
import gc
from copy import deepcopy
#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git

from  glm_utils import tuple_type
class NeuralDGLM(pl.LightningModule, GLMMixin):
    
    glm_type = "DGLM"

    def __init__(self,
            scaler_features:Union[MinMaxScaler,StandardScaler],
            scaler_target:Union[MinMaxScaler,StandardScaler],

            nn_name:str=None,
            nn_params:Dict={},
            target_distribution_name = 'lognormal_hurdle',
            
            mu_distribution_name='normal',
            mu_link_name='identity',
            mu_params=None,
            
            dispersion_distribution_name='gamma',
            dispersion_link_name='negative_inverse',
            disp_params=None,

            p_link_name='sigmoid',
            p_params = None, 

            pos_weight=1,
            save_hparams=True,
            debugging = False,

            min_rain_value= 1.0,
            task = None,
            beta1 = None,
            
            **kwargs):

        super().__init__()

        # Task
        self.task = task
        self.dconfig = kwargs.get('dconfig', None)

        # Load Neural Model
        neural_net_class = MAP_NAME_NEURALMODEL[nn_name]
        neural_net = neural_net_class( **nn_params )
        self.neural_net = neural_net

        # Saving specific hyper-parameters to hparams file
        # this should only run if you are about to train model, atm it runs whenever you load model
        ignore_list = ['save_hparams', "scaler_features","scaler_target","debugging","feature_start_date",
                "train_start_date","test_set_size_elements","train_set_size_elements",
                "val_set_size_elements","original_uk_dim"]
        if save_hparams:
            self.save_hyperparameters(ignore=ignore_list)

        # Target Distribution
        self.target_distribution_name = target_distribution_name
        self.target_distribution = self._get_distribution( self.target_distribution_name )() #target distribution
        self.loss_fct = self._get_loglikelihood_loss_func( target_distribution_name )( pos_weight=pos_weight, **kwargs )  #loss function

        # Checking compatibility of target distribution and link functions
        # assert self.check_distribution_link(mu_distribution_name, mu_link_name), "Incompatible mu link function chosen for target distribution"
        # assert self.check_distribution_link(dispersion_distribution_name, dispersion_link_name),  "Incompatible dispersion link function chosen for target distribution"

        # mu term distribution/link name / inverse_link function
        self.mu_distribution_name = mu_distribution_name
        self.mu_link_name = mu_link_name
        self.mu_inverse_link_function = self._get_inv_link(self.mu_link_name, mu_params)
        
        # Dispersion term distribution/link name / inverse_link function
        self.dispersion_distribution_name = dispersion_distribution_name
        self.dispersion_link_name =  dispersion_link_name
        self.dispersion_inverse_link_function  = self._get_inv_link(self.dispersion_link_name, disp_params)

        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.register_buffer('target_scale'  , torch.as_tensor( scaler_target.scale_[0]) )
        self.register_buffer('min_rain_value', torch.as_tensor( min_rain_value) )

        # This mu and max needs to be dependent on the scaler
        self.min_dispersion_output_standardized, self.max_disperion_output_standardized = self._get_dispersion_range( distribution_name=self.target_distribution_name )

        # Setting on hurdle neural_net 
        self.p_link_name = p_link_name
        self.p_inverse_link_function = self._get_inv_link(self.p_link_name, p_params)

        # optimizer params
        self.beta1 = beta1
        
        # reference to logger in case of debugging
        self.pixel_sample_prop = kwargs.get('pixel_sample_prop', None)
        self.debugging = debugging
        
    def forward(self, x ):
        
        output = self.neural_net(x)
        
        #Applying mean function
        mu = self.mu_inverse_link_function(output['mu'])
        disp = self.dispersion_inverse_link_function(output['disp'])
        p = self.p_inverse_link_function(output['logits']) 

        #clamp dispersion term
        disp = disp.clone()
        with torch.no_grad():
            if self.min_dispersion_output_standardized or self.max_disperion_output_standardized:
                disp.clamp_(self.min_dispersion_output_standardized, self.max_disperion_output_standardized)

        mu = mu.squeeze(-1)
        disp = disp.squeeze(-1)
        p = p.squeeze(-1)
        output['logits'] = output['logits'].squeeze(-1)
    
        output['mu'] = mu
        output['disp'] = disp
        output['p'] = p
        
        return output

    def step(self, batch, step_name ):

        #For the case of UK Eobs dset
        inp = batch['input']
        target_rain_value_scaled = batch['target']
        target_did_rain = torch.where( target_rain_value_scaled > 0.5*self.target_scale, 1.0, 0.0 )
        mask = batch.get('mask',None)
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.to(torch.bool)
            mask = ~mask
            
        idx_loc_in_region = batch.get('idx_loc_in_region',None)
        
        # Predicting
        output  = self.forward(inp)

        pred_mu_scaled = output['mu']
        pred_disp_scaled = output['disp'] 
        pred_p_scaled  = output['p']
        pred_logits_scaled = output['logits']

        # extracting the central region of interest
        bounds = Era5EobsDataset.central_region_bounds(self.dconfig) #list [ lower_h_bound[0], upper_h_bound[0], lower_w_bound[1], upper_w_bound[1] ]

        if pred_mu_scaled.dim() == 2 :
            pred_mu_scaled = pred_mu_scaled[..., None, None]
            pred_disp_scaled = pred_disp_scaled[..., None, None]
            pred_p_scaled = pred_p_scaled[..., None, None]
            pred_logits_scaled = pred_logits_scaled[..., None, None]
            target_rain_value_scaled = target_rain_value_scaled[..., None, None]
            target_did_rain = target_did_rain[..., None, None]
            mask = mask[..., None, None]
            
        pred_mu_scaled   = Era5EobsDataset.extract_central_region(pred_mu_scaled, bounds )
        pred_disp_scaled   = Era5EobsDataset.extract_central_region(pred_disp_scaled, bounds )
        pred_p_scaled   = Era5EobsDataset.extract_central_region(pred_p_scaled, bounds )
        pred_logits_scaled   = Era5EobsDataset.extract_central_region(pred_logits_scaled, bounds )
        mask    = Era5EobsDataset.extract_central_region(mask, bounds )
        target_did_rain = Era5EobsDataset.extract_central_region(target_did_rain, bounds )
        target_rain_value_scaled  = Era5EobsDataset.extract_central_region(target_rain_value_scaled, bounds )

        # Patch/Pixel masking loss 
        pixel_mask = True #Defaults to True
        if pred_mu_scaled.shape[-1]!=1 or pred_mu_scaled.shape[-2]!=1:
            #No pixel masking if we have a point prediction
            pixel_mask = torch.bernoulli( torch.empty( pred_mu_scaled.shape, device=mask.device ).fill_(self.pixel_sample_prop)  )
            pixel_mask = pixel_mask.to(torch.bool)

        # applying mask for loss and evaluation metrics
        pred_mu_scaled_masked      = torch.masked_select(pred_mu_scaled, (mask & pixel_mask ) )
        pred_disp_scaled_masked    = torch.masked_select(pred_disp_scaled, (mask & pixel_mask) ) 
        pred_p_scaled_masked       = torch.masked_select(pred_p_scaled, (mask & pixel_mask) ) 
        target_rain_value_scaled_masked   = torch.masked_select(target_rain_value_scaled, (mask & pixel_mask) )
        pred_logits_scaled_masked = torch.masked_select(pred_logits_scaled, (mask & pixel_mask) )

        target_did_rain_masked = torch.masked_select(target_did_rain, (mask & pixel_mask) )
    
        # Calculating Losses
        loss, composite_losses = self.loss_fct( target_rain_value_scaled_masked, target_did_rain_masked, pred_mu_scaled_masked, 
                                    pred_disp_scaled_masked, logits=pred_logits_scaled_masked,
                                    p=pred_p_scaled_masked, global_step=self.global_step)

        #Unscaling predictions for prediction metrics
        pred_mu_masked, pred_disp_masked, pred_p_masked = self.target_distribution.unscale_distribution( 
                                                    pred_mu_scaled_masked.detach(), pred_disp_scaled_masked.detach(),
                                                    pred_p_scaled_masked.detach(), self.scaler_target )

        pred_mean_masked = self.target_distribution.get_mean( pred_mu_masked, pred_disp_masked, pred_p_masked )

        target_rain_value_masked = self.unscale_rain(target_rain_value_scaled_masked, self.scaler_target)

        pred_metrics = self.loss_fct.prediction_metrics(target_rain_value_masked, target_did_rain_masked, pred_mean_masked,
                                                            logits=pred_logits_scaled_masked, 
                                                            p=pred_p_masked, min_rain_value=self.min_rain_value )
        
        loss.masked_fill_(loss.isnan(), 0)

        # Logging
        if self.debugging and step_name=='train' and pred_mu_masked.numel() != 0:
            tblogger = self.trainer.logger.experiment
            global_step = self.trainer.global_step
        
            tblogger.add_histogram('mu', pred_mu_masked, global_step=global_step)
            tblogger.add_histogram('disp', pred_disp_masked, global_step=global_step)        
            tblogger.add_histogram('p', pred_p_masked, global_step=global_step )

        if step_name in ['train','val']:
            return {'loss':loss, 'composite_losses':composite_losses, 'pred_metrics':pred_metrics }

        elif step_name in ['test']:
            # Scaling / Masking but also maintaining tensor shape

            pred_mu, pred_disp, pred_p = self.target_distribution.unscale_distribution( 
                                                        pred_mu_scaled.detach(), pred_disp_scaled.detach(),
                                                        pred_p_scaled.detach(), self.scaler_target)
            target_rain_value = self.unscale_rain(target_rain_value_scaled, self.scaler_target)

            output =  {'loss':loss,
                       
                       'pred_mu':pred_mu.detach().to('cpu'), 
                        'pred_disp':pred_disp.detach().to('cpu'),  
                        'pred_p':pred_p.detach().to('cpu'),
                        
                        'target_did_rain':target_did_rain.detach().to('cpu'),
                        'target_rain_value':target_rain_value.detach().to('cpu'), 

                        'pred_metrics':pred_metrics,
                        'composite_losses':composite_losses,
                        
                        'mask':mask if mask is None else mask.detach().to('cpu'),
                        'idx_loc_in_region':idx_loc_in_region.detach().to('cpu') if self.task == "uk_rain" else None

                            }
                
            return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        output = self.step(batch, "train")
        self.log("train_loss/loss",output['loss'])

        if output.get( 'composite_losses', None):
            self.log("train_loss/norain",output['composite_losses']['loss_norain'])
            self.log("train_loss/rain",output['composite_losses']['loss_rain'])
        
        if output.get( 'pred_metrics', None):
            if output['pred_metrics']['pred_acc']: self.log("train_metric/acc",output['pred_metrics']['pred_acc'],  on_step=False, on_epoch=True )
            if output['pred_metrics']['pred_rec']: self.log("train_metric/recall",output['pred_metrics']['pred_rec'], on_step=False, on_epoch=True )
            if output['pred_metrics']['pred_mse']: self.log("train_metric/mse_rain",output['pred_metrics']['pred_mse'],  on_step=False, on_epoch=True )
            if output['pred_metrics']['pred_r10mse']: 
                if not torch.isnan( output['pred_metrics']['pred_r10mse'] ).any():
                    self.log("train_metric/r10mse_rain",output['pred_metrics']['pred_r10mse'] , on_step=False, on_epoch=True)


        return output
    
    def validation_step(self, batch, batch_idx):
        output  = self.step(batch, "val")
        output["val_loss/loss"] = output["loss"]
        self.log("val_loss/loss", output['loss'], prog_bar=True)

        if output.get('composite_losses', None):
            self.log("val_loss/norain", output['composite_losses']['loss_norain'], on_step=False, on_epoch=True)
            self.log("val_loss/rain", output['composite_losses']['loss_rain'], on_step=False, on_epoch=True)

        if output.get( 'pred_metrics', None):
            if output['pred_metrics']['pred_acc']: self.log("val_metric/acc",output['pred_metrics']['pred_acc'] , on_step=False, on_epoch=True)
            if output['pred_metrics']['pred_rec']: self.log("val_metric/recall",output['pred_metrics']['pred_rec'] , on_step=False, on_epoch=True)
            if output['pred_metrics']['pred_mse']: self.log("val_metric/mse_rain",output['pred_metrics']['pred_mse'],  on_step=False, on_epoch=True )
            if output['pred_metrics']['pred_r10mse']: 
                if not torch.isnan( output['pred_metrics']['pred_r10mse'] ).any():
                    self.log("val_metric/r10mse_rain",output['pred_metrics']['pred_r10mse'] , on_step=False, on_epoch=True)
        return output

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        res = super().on_save_checkpoint(checkpoint)
        try:
            mc = next( filter( lambda cb: isinstance(cb, ModelCheckpoint), self.trainer.callbacks) )
            if not os.path.exists(mc.dirpath): os.makedirs(mc.dirpath, exist_ok=True)
            mc.to_yaml()
        except FileNotFoundError as e:
            pass

        return res

    def test_step(self, batch, batch_idx):

        output = self.step(batch, "test")

        # Logging the aggregate loss during testing
        self.log("test_loss/loss", output['loss'])

        if output.get('composite_losses', None):
            if not output['composite_losses']['loss_norain'].isnan().any():
                self.log("test_loss/norain", output['composite_losses']['loss_norain'], on_epoch=True, prog_bar=True)

            if not output['composite_losses']['loss_rain'].isnan().any():
                self.log("test_loss/rain", output['composite_losses']['loss_rain'], on_epoch=True, prog_bar=True)

        if output.get('pred_metrics', None):
            if output['pred_metrics']['pred_acc']: self.log("test_metric/acc",output['pred_metrics']['pred_acc'] , on_step=False, on_epoch=True)
            if output['pred_metrics']['pred_rec']: self.log("test_metric/recall",output['pred_metrics']['pred_rec'] , on_step=False, on_epoch=True)
            if output['pred_metrics']['pred_mse']: self.log("test_metric/mse_rain",output['pred_metrics']['pred_mse'] , on_step=False, on_epoch=True )
            if output['pred_metrics']['pred_r10mse']:
                if not torch.isnan( output['pred_metrics']['pred_r10mse'] ).any().item():
                    self.log("test_metric/r10mse_rain",output['pred_metrics']['pred_r10mse'], on_step=False, on_epoch=True, )

        if self.task == "uk_rain": 
            output['li_locations'] = batch.pop('li_locations',None)
            output['target_date_window'] = batch.pop('target_date_window',None)
        
        # downcasting       
        output['pred_mu'] = output['pred_mu'].numpy().astype(np.float32)
        output['pred_disp'] = output['pred_disp'].numpy().astype(np.float32)
        output['pred_p'] = output['pred_p'].numpy().astype(np.float16)
        output['target_did_rain'] = output['target_did_rain'].numpy().astype(np.float16)
        output['target_rain_value'] = output['target_rain_value'].numpy().astype(np.float16)
        output['idx_loc_in_region'] = output['idx_loc_in_region'].numpy().astype(np.float16)
        output['mask'] = output['mask'].numpy()
        
        if output['pred_metrics']['pred_mse'] is not None:
            output['pred_metrics']['pred_mse'] = output['pred_metrics']['pred_mse'].to('cpu').numpy().astype(np.float16)
        
        if output['pred_metrics']['pred_rec'] is not None:
            output['pred_metrics']['pred_rec'] = output['pred_metrics']['pred_rec'].to('cpu').numpy().astype(np.float16)
        
        if output['pred_metrics']['pred_acc'] is not None:
            output['pred_metrics']['pred_acc'] = output['pred_metrics']['pred_acc'].to('cpu').numpy().astype(np.float16)
        
        if output['pred_metrics']['pred_r10mse'] is not None:
            output['pred_metrics']['pred_r10mse'] = output['pred_metrics']['pred_r10mse'].to('cpu').numpy().astype(np.float16) if output['pred_metrics']['pred_r10mse'] else np.nan
        
        output.pop('composite_losses',None)
        output.pop('loss')
        
        return output

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        print("\nSaving Test Output to File")
    
        # Saving the predictions and targets from the test sets
        pred_mu = np.concatenate( [output.pop('pred_mu')  for output in outputs], axis=0)
        pred_disp = np.concatenate( [output.pop('pred_disp') for output in outputs], axis=0)
        pred_p = np.concatenate( [output.pop('pred_p') for output in outputs], axis=0)
        gc.collect()
        idx_loc_in_region = np.concatenate( [output.pop('idx_loc_in_region') for output in outputs], axis=0)


        #target data
        target_did_rain = np.concatenate([output.pop('target_did_rain') for output in outputs], axis=0 )
        target_rain_value = np.concatenate([output.pop('target_rain_value') for output in outputs], axis=0 )
        target_date_window = np.concatenate([output.pop('target_date_window') for output in outputs] )
        gc.collect()
        # Split predictions by location
        test_dl = self.trainer.test_dataloaders[0]


        dconfig = test_dl.dataset.dconfig
                    
        dict_loc_data = defaultdict(dict)
        
        mask = np.concatenate( [ output.pop('mask') for output in outputs], axis=0 )
        
        
        #Flattening all the objects
        pred_mu = pred_mu.reshape(-1, *pred_mu.shape[2:] )
        pred_disp = pred_disp.reshape(-1, *pred_disp.shape[2:] )
        pred_p = pred_p.reshape(-1, *pred_p.shape[2:] )
        idx_loc_in_region = idx_loc_in_region.repeat(dconfig.lookback_target,0)
        target_did_rain =target_did_rain.reshape(-1, *target_did_rain.shape[2:] )
        target_rain_value = target_rain_value.reshape(-1, *target_rain_value.shape[2:] )
        mask = mask.reshape(-1, *mask.shape[2:] )
        target_date_window = target_date_window.reshape(-1 )
        
        _ = np.concatenate( [output.pop('li_locations') for output in outputs], axis=0 )
        locations = _.reshape(-1)
        
        gc.collect()
        # === Determining the start and end index for each location's portion of the dataset   

        # Assuming we consider predictions in blocks of 7 (lookback) the below marks every seventh block of predictions where i is where new data for where location starts
        # Gathering indexes indicating where predictions should be split since the location has shifted
        start_idxs_for_location_subset = [ (idx,str(loc)) for idx,loc in enumerate(locations) if (idx==0 or loc!=locations[idx-1]) ]
        end_idxs_for_location_subset = start_idxs_for_location_subset[1:] + [ (len(locations), "End") ] 

        assert end_idxs_for_location_subset[-1][0] == pred_mu.shape[0], "The final end idxs should be the same as the length of the overall dataset"
                    
        dict_loc_li_data = defaultdict(dict)
        
        for idx  in range( len( start_idxs_for_location_subset)-1 ):

            loc = start_idxs_for_location_subset[idx][1]
            s_idx = start_idxs_for_location_subset[idx][0]
            e_idx = end_idxs_for_location_subset[idx][0]

            # Need to concat data for each city
            datum_start_date = dconfig.test_start if (loc not in dict_loc_data.keys()) else ( dict_loc_data[loc]['date'][-1]+pd.to_timedelta(1,'D') )

            data = {
                'pred_mu':pred_mu[s_idx:e_idx],
                    'pred_disp':pred_disp[s_idx:e_idx],
                    'target_did_rain':target_did_rain[s_idx:e_idx],
                    'target_rain_value':target_rain_value[s_idx:e_idx],
                    'date':target_date_window[s_idx:e_idx],
                    'pred_p': pred_p[s_idx:e_idx],
                    'mask':mask[s_idx:e_idx],
                    'idx_loc_in_region': idx_loc_in_region[s_idx:e_idx] }


            # For each location store  a list of the data predictions
            if loc not in dict_loc_li_data:
                dict_loc_li_data[loc] = [data]
            else:
                dict_loc_li_data[loc].append(data)
            # del data
        
                    
        # Now concatenating all the predictions along time axis for each location
        # Also sorting predictions
        feature_names = deepcopy(list(next( iter(dict_loc_li_data[loc] )).keys()))
        unique_locations = deepcopy(list(dict_loc_li_data.keys()))
        
        for loc in unique_locations:
            
            li_data = dict_loc_li_data.pop(loc)
            
            li_dates = np.concatenate( tuple( d['date'] for d in  li_data ) ) 
            ordered_idxs = np.argsort( li_dates )
            
            for k in feature_names:
                                    
                dict_loc_data[loc][k] = np.concatenate( tuple(d.pop(k) for d in li_data ) ) [ordered_idxs]
                                
            del li_data
            gc.collect()
                                
        dir_path = os.path.dirname( os.path.dirname( self.trainer.resume_from_checkpoint ) )  if self.trainer.resume_from_checkpoint else self.trainer.log_dir

        #add teststart_testend to end of test_output.pkl and summary.json fns
        suffix = f"{dconfig.test_start}_{dconfig.test_end}"
        file_path = os.path.join( dir_path, f"test_output_{suffix}.pkl" ) 
        with open( file_path, "wb") as f:
            pickle.dump( dict_loc_data, f )
        
        #TODO {rilwan.ade} clean up this implementation
        
        try:    
            test_mse =  np.stack( [output['pred_metrics'].pop('pred_mse')  for output in outputs if output['pred_metrics']['pred_mse'] ], axis=0 ).mean().squeeze().item()  
        except Exception:
            test_mse = np.nan
        try:
            test_r10mse =  np.nanmean(np.stack( [output['pred_metrics'].pop('pred_r10mse') for output in outputs if output['pred_metrics']['pred_r10mse']], axis=0 )).squeeze().item() 
        except Exception:
            test_r10mse = np.nan
        try:    
            test_acc = np.stack( [output['pred_metrics'].pop('pred_acc') for output in outputs if output['pred_metrics']['pred_acc']], axis=0 ).mean().squeeze().item()
        except Exception:
            test_acc = np.nan
        
        try:
            test_rec = np.stack( [output['pred_metrics'].pop('pred_rec') for output in outputs if output['pred_metrics']['pred_rec']], axis=0 ).mean().squeeze().item()
        except Exception:
            test_rec = np.nan
            
        # Recording losses on test set and summarised information about test run
        summary = {
            'train_start': dconfig.train_start,
            'train_end':dconfig.train_end,
            'val_start':dconfig.val_start,
            'val_end':dconfig.val_end,
            'test_start':dconfig.test_start,
            'test_end':dconfig.test_end,
            'test_mse':  test_mse  ,
            
            'test_r10mse':  test_r10mse,
            'test_acc': test_acc,
            'test_rec': test_rec ,
        }

        file_path_summary = os.path.join(dir_path, f"summary_{suffix}.json")
        with open(file_path_summary, "w") as f:
            json.dump( summary, f)

        return super().test_epoch_end(outputs)
        
    def configure_optimizers(self):
        #debug
        optimizer = Adafactor(self.parameters(), scale_parameter=True,
                                relative_step=True, 
                                warmup_init=True,
                                lr=None, 
                                beta1=self.beta1,
                                clip_threshold=0.5)
        
        lr_scheduler = AdafactorSchedule(optimizer)
        # lr_scheduler = get_constant_schedule_with_warmup( optimizer, num_warmup_steps=1000, last_epoch=-1)
        
        return { 'optimizer':optimizer, 
                    'lr_scheduler':lr_scheduler,
                    'frequency':1,
                     }

    def save_scalers(self, dir_path:str, **kwargs ) -> None:
        os.makedirs(dir_path, exist_ok=True)
        if hasattr(self, 'scaler_features'):
            pickle.dump(self.scaler_features, open(os.path.join(dir_path, "scaler_features.pkl"),"wb") )

        if hasattr(self, 'scaler_target'):
            pickle.dump(self.scaler_target, open(os.path.join(dir_path, "scaler_target.pkl"),"wb") )

    @classmethod
    def load_scalers(cls, dir_path:str=None):
        """Loads the scaler object from a given path.
                The path can be a path to the directory containing both the feature and target scaler

        Args:
            dir_path (str, optional): [description]. Defaults to None.


        Returns:
            [type]: [description]
        """

        path_scaler_features = os.path.join(dir_path, "scaler_features.pkl")
        path_scaler_target = os.path.join(dir_path, "scaler_target.pkl")

        if os.path.exists(path_scaler_features) and os.path.exists(path_scaler_target):
            scaler_features = pickle.load(open(path_scaler_features,"rb"))
            scaler_target = pickle.load(open(path_scaler_target,"rb"))
        else:
            raise FileNotFoundError(f"The feature and target scalers can not be found at the directory below:\n {dir_path}") 
        
        return scaler_features, scaler_target

    @staticmethod
    def parse_glm_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument("--target_distribution_name", default="compound_poisson")

        parser.add_argument("--beta1", type=float, default=0.2 )

        parser.add_argument("--mu_distribution_name", default="normal")
        parser.add_argument("--mu_link_name", default="identity",help="name of link function used for mu distribution")
        parser.add_argument("--mu_params", default=None, type=tuple_type)

        parser.add_argument("--dispersion_distribution_name", default="gamma")
        parser.add_argument("--dispersion_link_name", default="relu",help="name of link function used for mu distribution")
        parser.add_argument("--disp_params", default=None, type=tuple_type)

        parser.add_argument("--p_link_name", default="sigmoid")
        parser.add_argument("--p_params", default=None, type=tuple_type)



        parser.add_argument("--pos_weight", default=1.2, type=float ,help="The relative weight placed on examples where rain did occur when calculating the loss")
        parser.add_argument("--pixel_sample_prop", default=0.7, type=float ,help="")

        # Compound Poisson arguments
        parser.add_argument('--cp_version', default=None, type=int)
        parser.add_argument('--max_j', default=None, type=int)
        parser.add_argument('--j_window_size',default=None, type=int)
        parser.add_argument('--target_range',default=(0,4), type=tuple_type)


        glm_args = parser.parse_known_args()[0]
        return glm_args

    @staticmethod
    def get_ckpt_path(_dir_checkpoint, mode='best'):
        if mode == 'best':
            checkpoint_yaml_file = os.path.join(_dir_checkpoint, "best_k_models.yaml")
            # key= ckptpath, value = val_loss
            scores_dict = yaml.load(open(checkpoint_yaml_file, "r"), Loader=yaml.FullLoader)
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False: 
                    
                best_ckpt_path = os.path.abspath( os.path.join(
                    Path(_dir_checkpoint).parents[4],
                    best_ckpt_path) )

            path = best_ckpt_path

        else:
            raise NotImplementedError

        return path

    
MAP_NAME_GLM = {'DGLM':NeuralDGLM}
