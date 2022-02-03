from typing import Union
import torch
from torch.autocast_mode import autocast
# from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
import pickle
import os
from better_lstm import LSTM
import einops
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import torchtyping
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git


class NeuralDGLM(pl.LightningModule, GLMMixin ):
    
    glm_type = "DGLM"

    def __init__(self,
            neural_net:nn.Module,
            target_distribution_name = 'lognormal_hurdle',
            
            mean_distribution_name='normal',
            mean_link_name='identity',
            mean_link_func_params={},
            
            dispersion_distribution_name='gamma',
            dispersion_link_name='negative_inverse',
          

            scaler_features:Union[MinMaxScaler,StandardScaler,str]=None,
            scaler_targets:Union[MinMaxScaler,StandardScaler,str]=None,

            pos_weight=1,
            **kwargs):

        super().__init__()

        # Saving specific hyper-parameters to hparams file
        ignore_list = []
        if not type(ignore_list)==str: ignore_list.append("neural_net")
        if not type(ignore_list)==str: ignore_list.append("scaler_features") 
        if not type(ignore_list)==str: ignore_list.append("scaler_targets") 
        self.save_hyperparameters(ignore=ignore_list)

        self.neural_net = neural_net

        # Target Distribution
        self.target_distribution_name = target_distribution_name
        self.target_distribution = self._get_distribution( self.target_distribution_name ) #target distribution
        self.loss_fct = self._get_loglikelihood_loss_func( target_distribution_name )( pos_weight=pos_weight, **kwargs )  #loss function

        # Checking compatibility of target distribution and link functions
        assert self.check_distribution_mean_link(target_distribution_name, mean_link_name), "Incompatible mean link function chosen for target distribution"
        assert self.check_distribution_dispersion_link(target_distribution_name, dispersion_link_name),  "Incompatible dispersion link function chosen for target distribution"


        # Mean term distribution/link name / inverse_link function
        self.mean_distribution_name = mean_distribution_name
        self.mean_link_name = mean_link_name
        self.mean_inverse_link_function = self._get_inv_link(self.mean_link_name, **mean_link_func_params)
        
        # Dispersion term distribution/link name / inverse_link function
        self.dispersion_distribution_name = dispersion_distribution_name
        self.dispersion_link_name =  dispersion_link_name
        self.dispersion_inverse_link_function  = self._get_inv_link(self.dispersion_link_name)

        # Restraints on predictions for MaxMin standardized target
        if isinstance(scaler_features,str) and isinstance(scaler_targets,str):
            scaler_features, scaler_targets = NeuralDGLM.load_scalers(None, scaler_features, scaler_targets)

        self.scaler_features = scaler_features
        self.scaler_targets = scaler_targets

        # This mean and max needs to be dependent on the scaler
        self.min_mean_output_standardized, self.max_mean_output_standardized = self._get_mean_range( self.target_distribution_name, self.scaler_targets ) 
        self.min_dispersion_output_standardized, self.max_disperion_output_standardized = self._get_dispersion_range( distribution_name=self.target_distribution_name )

        # Setting on hurdle neural_net 
        if self.neural_net.p_variable_model:
            self.p_inverse_link_function = self._get_p_inverse_link_function(target_distribution_name)
            
    def forward( self, x ):
        output = self.neural_net(x)

        mean = self.mean_inverse_link_function(output['mean'])
        disp = self.dispersion_inverse_link_function(output['disp'])
        if self.neural_net.p_variable_model:
            output['p'] = self.p_inverse_link_function(output['logits']) 
          
        mean = mean.clone()
        disp = disp.clone()

        with torch.no_grad():
            mean.clamp_(self.min_mean_output_standardized, self.max_mean_output_standardized)            
            disp.clamp_(self.min_dispersion_output_standardized, self.max_disperion_output_standardized)

        mean = mean.squeeze(-1)
        disp = disp.squeeze(-1)
    
        output['mean'] = mean
        output['disp'] = disp
        
        return output

    def step(self, batch, step_name ):
        
        inp, target = batch
        target_rain_bool, target_rain_value = torch.unbind(target,-1)

        output  = self.forward(inp)
        pred_mean = output['mean']
        pred_disp = output['disp']  
         
        # during initial training steps fix the dispersion term to be within a specific range until stability is reached
    
        if self.neural_net.p_variable_model:
            p = output['p'].squeeze(-1)  
            logits = output['logits'].squeeze(-1)     
            loss, composite_losses = self.loss_fct( target_rain_value, target_rain_bool , pred_mean, pred_disp, logits=logits, p=p )
        else:
            loss = self.loss_fct( target_rain_value, pred_mean, pred_disp )
            composite_losses = None

        if step_name in ['train','val']:
            return {'loss':loss, 'composite_losses':composite_losses}

        elif step_name in ['test']:
            output =  {'loss':loss, 'pred_mean':pred_mean, 'pred_disp':pred_disp,  
                            'target_rain_bool':target_rain_bool, 'target_rain_value':target_rain_value }

            if self.neural_net.p_variable_model:
                output['composite_losses'] = composite_losses
                output['pred_p'] = p
            return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        output = self.step(batch, "train")
        self.log("loss",output['loss'])

        if output.get( 'composite_losses', None):
            self.log("loss\norain",output['composite_losses']['loss_norain'])
            self.log("loss\rain",output['composite_losses']['loss_rain'])

        return output
    
    def validation_step(self, batch, batch_idx):
        output  = self.step(batch, "val")
        self.log("val_loss", output['loss'], prog_bar=True)
        if output.get('composite_losses', None):
            self.log("val_loss\norain", output['composite_losses']['loss_norain'], on_step=False, on_epoch=True)
            self.log("val_loss\rain", output['composite_losses']['loss_rain'], on_step=False, on_epoch=True)

        return output

    def test_step(self, batch, batch_idx):

        output = self.step(batch, "test")

        # Logging the aggregate loss during testing
        self.log("test_loss", output['loss'])
        self.log("test_loss\norain", output['composite_losses']['loss_norain'], on_epoch=True, prog_bar=True)
        self.log("test_loss\rain", output['composite_losses']['loss_rain'], on_epoch=True, prog_bar=True)
        
        return output

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        
        # Saving the predictions and targets from the test sets
        pred_mean = torch.cat( [output['pred_mean'] for output in outputs], dim=0 ).cpu().numpy()
        pred_disp = torch.cat( [output['pred_disp'] for output in outputs], dim=0 ).cpu().numpy()

        if self.neural_net.p_variable_model:
            pred_p = torch.cat( [output['pred_p'] for output in outputs], dim=0 ).cpu().numpy()
        else:
            pred_p = None
        
        target_rain_bool = torch.cat( [output['target_rain_bool'] for output in outputs], dim=0 ).cpu().numpy()
        target_rain_value = torch.cat( [output['target_rain_value'] for output in outputs], dim=0 ).cpu().numpy()
        
        pred_mean, pred_disp, p = self.destandardize( pred_mean, pred_disp, pred_p ,self.target_distribution_name, self.scaler_targets )

        # Split predictions by location
        test_dl = self.trainer.test_dataloaders[0]

        locations = [ ds.location for ds in test_dl.dataset.datasets]
        cumulative_sizes = [0] + test_dl.dataset.cumulative_sizes
        
        dict_location_data = {}
        for idx, loc in enumerate(locations):
            if cumulative_sizes==0:
                continue
            s_idx = cumulative_sizes[idx]
            e_idx = cumulative_sizes[idx+1]

            lookback = test_dl.dataset.datasets[idx].lookback
            dates = test_dl.dataset.datasets[idx].dates #dates in dataset that are valid
            indexes_filtrd = test_dl.dataset.datasets[idx].indexes_filtrd #indexes_filtrd
            
            date_windows = [ dates[idx-lookback:idx] for idx in indexes_filtrd[lookback: ] ]

            data = {'pred_mean':pred_mean[s_idx:e_idx],
                    'pred_disp':pred_disp[s_idx:e_idx],
                    'target_rain_bool':target_rain_bool[s_idx:e_idx],
                    'target_rain_value':target_rain_value[s_idx:e_idx],
                    'date':date_windows }
            
        
            if self.neural_net.p_variable_model:
                data['pred_p'] = pred_p[s_idx:e_idx]

            dict_location_data[loc] = data

        dir_path = os.path.dirname( next( ( callback for callback in self.trainer.callbacks if type(callback)==pl.callbacks.model_checkpoint.ModelCheckpoint) ).dirpath )
        file_path = os.path.join( dir_path, "test_output.pkl" ) 
        with open( file_path, "wb") as f:
            pickle.dump( dict_location_data, f )
        
        return super().test_epoch_end(outputs)
        
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None )
        lr_scheduler = AdafactorSchedule(optimizer)
        return { 'optimizer':optimizer, 
                    'lr_scheduler':lr_scheduler }

    def unstandardize_output(self, pred_mean):
        pred_mean = ( pred_mean * self.scaler_targets.scale_ ) + self.scaler_targets.mean_
        return pred_mean  

    def save_scalers(self, dir_path:str ) -> None:
        if hasattr(self, 'scaler_features'):
            pickle.dump(self.scaler_features,open(os.path.join(dir_path,"scaler_features.pkl"),"wb") )

        if hasattr(self, 'scaler_targets'):
            pickle.dump(self.scaler_features,open(os.path.join(dir_path,"scaler_targets.pkl"),"wb") )
    
    @staticmethod
    def load_scalers(dir_path:str=None, path_scaler_features:str=None, path_scaler_targets:str=None ):

        assert bool(dir_path) ^ ( bool(path_scaler_features) and bool(path_scaler_targets) ) 

        scaler_features, scaler_targets = None

        if not path_scaler_features:
            path_scaler_features = os.path.join(dir_path, "scaler_features.pkl")
        if not path_scaler_targets:
            path_scaler_targets = os.path.join(dir_path, "scaler_targets.pkl")

        if os.path.exists(path_scaler_features):
            scaler_features = pickle.load(open(path_scaler_features,"rb"))
        if os.path.exists(path_scaler_targets):
            scaler_targets = pickle.load(open(path_scaler_targets,"rb"))

        return scaler_features, scaler_features

    @staticmethod
    def parse_glm_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument("--target_distribution_name", default="compound_poisson")

        parser.add_argument("--mean_distribution_name", default="normal")
        parser.add_argument("--mean_link_name", default="identity",help="name of link function used for mean distribution")

        parser.add_argument("--dispersion_distribution_name", default="gamma")
        parser.add_argument("--dispersion_link_name", default="relu",help="name of link function used for mean distribution")

        parser.add_argument("--pos_weight", default=2, type=int ,help="The relative weight placed on examples where rain did occur when calculating the loss")

        # Compound Poisson arguments
        parser.add_argument('--cp_version', default=None, type=int)
        parser.add_argument('--max_j', default=None, type=int)
        parser.add_argument('--j_window_size',default=None, type=int)

        glm_args = parser.parse_known_args()[0]
        return glm_args

MAP_NAME_GLM = {'DGLM':NeuralDGLM}
