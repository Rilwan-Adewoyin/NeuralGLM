import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
import pickle
import os
from loss_utils import LossMixin
from better_lstm import LSTM
import einops
#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git


class MLP(nn.Module):

    model_type = "MLP"

    def __init__(self, input_shape=(6,), output_shape=(1,), **kwargs ):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.encoder = nn.Sequential(nn.Linear( *self.input_shape, 6), 
                                nn.SELU() ,
                                nn.Linear(6, 6),
                                nn.SELU() ) 

        self.outp_mean = nn.Linear(6,*output_shape)
        self.outp_dispersion =  nn.Linear(6, *self.output_shape)

    def forward(self, x):
        h = self.encoder(x)
        pred_mean = self.outp_mean(h)
        pred_dispersion = self.outp_dispersion(h)
        return pred_mean, pred_dispersion

class HLSTM(nn.Module):

    model_type = "HLSTM"

    def __init__(self, input_shape=(6,), output_shape=(2,), hidden_dim=64, num_layers=3, hurdle_model=False, zero_inflated_model=False ) -> None:
        super().__init__()
    
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hurdle_model = hurdle_model

        self.encoder = nn.Sequential(
            nn.Linear( input_shape[0], hidden_dim),
            nn.SELU(),
            LSTM( input_size = hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                    dropouti=0.25, dropoutw=0.25, dropouto=0.25,
                    bidirectional=True),
            ExtractLSTMOutputFeatures()
        )

        if hurdle_model:
            self.outp_probrain = nn.Linear(hidden_dim*2, *self.output_shape)

        self.outp_mean = nn.Linear(hidden_dim*2,*self.output_shape)
        self.outp_dispersion =  nn.Linear(hidden_dim*2, *self.output_shape) 
        
    # @tor.autocast()
    def forward(self, x, standardized_output=True):
        h = self.encoder(x)
        output = {}
        if self.hurdle_model:
            output['prob'] = self.outp_probrain(h)

        output['mean'] = self.outp_mean(h)
        output['disp'] = self.outp_dispersion(h)
        
        return output
    
class ExtractLSTMOutputFeatures(nn.Module):

    def __init__(self, bidirectional=True, elem='all') -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.elem = elem

    def forward(self,x):
        out , _ = x

        # If Bi-Directional LSTM concatenating the Directions dimension
        if self.bidirectional == 2:
            out = einops.rearrange( out, 'b t d h -> b t (d h)')

        if self.elem == 'all':
            pass
        elif self.elem == 'last':
            out = out[:, -1:, :]

        return out
    

class NeuralDGLM(pl.LightningModule, GLMMixin, LossMixin ):
    
    glm_type = "DGLM"

    def __init__(self,
            model = "MLP",
            model_params = {},
            target_distribution_name = 'normal',
            
            mean_distribution_name='normal',
            mean_link_name='identity',
            mean_link_func_params={},
            dispersion_distribution_name='gamma',
            dispersion_link_name='negative_inverse',

            scaler_features=None,
            scaler_targets=None,
            **kwargs):

        super().__init__()
        self.save_hyperparameters()

        #neural network 
        if type(model)== str:
            self.model = MAP_NAME_NEURALMODEL[model](**model_params)
        else:
            self.model = model

        # Target Distribution
        self.target_distribution_name = target_distribution_name
        self.target_distribution = self._get_distribution( self.target_distribution_name )
        self.loss_fct = self._get_loglikelihood_loss_func( target_distribution_name )()

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
        self.scaler_features = scaler_features
        self.scaler_targets = scaler_targets
        self.min_mean_output_standardized, self.max_mean_output_standardized = self._get_mean_range( self.target_distribution_name, standardized=True ) 
        self.min_dispersion_output_standardized, self.max_disperion_output_standardized = self._get_dispersion_range( distribution_name=self.target_distribution_name, standardized=True )

        # Setting on hurdle model / zero-inflated model params
        if self.model.hurdle_model:
            self.prob_inverse_link_function = self._get_inv_link('sigmoid')
            
    def forward( self, x ):
        output = self.model(x)

        mean = self.mean_inverse_link_function(output['mean'])
        disp = self.dispersion_inverse_link_function(output['disp'])
          
        mean = mean.clone()
        disp = disp.clone()
        with torch.no_grad():
            mean.clamp_(self.min_mean_output_standardized, self.max_mean_output_standardized)
            disp.clamp_(self.min_dispersion_output_standardized, self.max_disperion_output_standardized)
        
        mean = mean.squeeze(-1)
        disp = disp.squeeze(-1)
        
        output['mean'] = mean
        output['disp'] = disp

        if self.model.hurdle_model:
            if not self.training:
                # We use BCEWithLogitsLoss during training, so no need for sigmoid transformation
                output['prob'] = self.prob_inverse_link_function(output['prob']) 

        return output

    def step(self, batch, step_name ):

        inp, target = batch
        target_rain_bool, target_rain_value = torch.unbind(target,-1)

        output  = self.forward(inp)
        pred_mean = output['mean']
        pred_disp = output['disp']  
         
        # during initial training steps fix the dispersion term to be within a specific range until stability is reached
        # if self.current_step < /2 :
        if self.model.hurdle_model:
            prob = output['prob'].squeeze(-1)     
            loss, composite_losses = self.loss_fct( target_rain_value, target_rain_bool , pred_mean, pred_disp, prob, use_logits=self.training )
        else:
            loss = self.loss_fct( target_rain_value, pred_mean, pred_disp )
            composite_losses = None

        if step_name in ['train','val']:
            return {'loss':loss, 'composite_losses':composite_losses}

        elif step_name in ['test']:
            output =  {'loss':loss, 'pred_mean':pred_mean, 'pred_disp':pred_disp,  
                            'target_rain_bool':target_rain_bool, 'target_rain_value':target_rain_value }

            if self.model.hurdle_model:
                output['composite_losses'] = composite_losses
                output['pred_prob'] = prob
            return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        output = self.step(batch, "train")
        self.log("loss",output['loss'])

        if output.get( 'composite_losses', None):
            self.log("loss\\bce",output['composite_losses']['loss_bce'])
            self.log("loss\cont",output['composite_losses']['loss_cont'])

        return output
    
    def validation_step(self, batch, batch_idx):
        output  = self.step(batch, "val")
        self.log("val_loss", output['loss'], prog_bar=True)
        if output.get('composite_losses', None):
            self.log("val_loss\\bce", output['composite_losses']['loss_bce'], on_step=False, on_epoch=True)
            self.log("val_loss\cont", output['composite_losses']['loss_cont'], on_step=False, on_epoch=True)

        return output

    def test_step(self, batch, batch_idx):

        output = self.step(batch, "test")

        # Logging the aggregate loss during testing
        self.log("test_loss",output['loss'] )
        self.log("test_loss\\bce", output['composite_losses']['loss_bce'], on_epoch=True, prog_bar=True )
        self.log("test_loss\cont",output['composite_losses']['loss_cont'], on_epoch=True ,prog_bar=True )
        
        return output

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        
        # Saving the predictions and targets from the test sets
        pred_mean = torch.cat( [output['pred_mean'] for output in outputs], dim=0 )
        pred_disp = torch.cat( [output['pred_disp'] for output in outputs], dim=0 )
        
        target_rain_bool = torch.cat( [output['target_rain_bool'] for output in outputs], dim=0 )
        target_rain_value = torch.cat( [output['target_rain_value'] for output in outputs], dim=0 )
        
        data = {'pred_mean':pred_mean.detach().numpy(),
                'pred_disp':pred_disp.detach().numpy(),
                'target_rain_bool':target_rain_bool.detach().numpy(),
                'target_rain_value':target_rain_value.detach().numpy() }
        
        if self.model.hurdle_model:
            pred_prob = torch.cat( [output['pred_prob'] for output in outputs], dim=0 )
            data['pred_prob'] = pred_prob
        
        dir_path = os.path.dirname( next( ( callback for callback in self.trainer.callbacks if type(callback)==pl.callbacks.model_checkpoint.ModelCheckpoint) ).dirpath )
        file_path = os.path.join( dir_path, "test_output.pkl" ) 
        with open( file_path, "wb") as f:
            pickle.dump( data, f )
        
        return super().test_epoch_end(outputs)
        
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None )
        lr_scheduler = AdafactorSchedule(optimizer)
        return { 'optimizer':optimizer, 
                    'lr_scheduler':lr_scheduler }

    def unstandardize_output(self, pred_mean):
        pred_mean = ( pred_mean * self.scaler_targets.scale_ ) + self.scaler_targets.mean_
        return pred_mean  

    
MAP_NAME_NEURALMODEL = {
    'MLP': MLP,
    'HLSTM': HLSTM
}
