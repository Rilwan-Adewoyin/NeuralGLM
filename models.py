import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
import pickle
import os
from loss_utils import LossMixin

class NeuralDGLM(pl.LightningModule, GLMMixin, LossMixin ):
    
    glm_type = "DGLM"

    def __init__(self,
            model = "MLP",
            model_params = {},
            target_distribution_name = 'normal',
            mean_distribution_name='normal',
            mean_link_name='identity',
            dispersion_distribution_name='gamma',
            dispersion_link_name='negative_inverse',
            **kwargs):

        super().__init__(**kwargs)

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
        self.mean_inverse_link_function = self._get_inv_link(self.mean_link_name)
        
        # Dispersion term distribution/link name / inverse_link function
        self.dispersion_distribution_name = dispersion_distribution_name
        self.dispersion_link_name =  dispersion_link_name
        self.dispersion_inverse_link_function  = self._get_inv_link(self.dispersion_link_name)

    def forward( self, x ):
        h = self.model(x)
        mean, disp = h
        return mean, disp

    def step(self, batch, step_name ):

        inp, target = batch

        h_mean , h_disp  = self.forward(inp)

        mean = self.mean_inverse_link_function(h_mean)
        disp = self.dispersion_inverse_link_function(h_disp)

        loss = self.loss_fct( target, mean, disp )

        if step_name in ['train','val']:
            return loss
        elif step_name in ['test']:
            return loss, mean, disp

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss = self.step(batch, "train")
        self.log("train_loss",loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss_mean  = self.step(batch, "val")
        self.log("val_loss",loss_mean,prog_bar=True)
        return loss_mean

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss_mean, mean, disp = self.step(batch, "test")
        self.log("test_loss",loss_mean)
        
        target = batch[1]
        pred_mean = mean
        pred_disp = disp

        return target, pred_mean, pred_disp
    
    def test_epoch_end(self, batch, batch_idx):

        target, pred_mean, pred_disp = zip(*batch)

        data = {'target':target.detach().numpy(),
                'pred_mean':pred_mean.detach().numpy(),
                'pred_disp':pred_disp.detach().numpy() }
        
        dir_path = f"./experiment_output/"
        os.makedirs(dir_path)
        with open( f"{dir_path}_{self.glm_type}_{self.model.model_type}_testoutput.pkl" , "rb") as f:
            pickle.dump( data, f )
        
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None )
        lr_scheduler = AdafactorSchedule(optimizer)
        return { 'optimizer':optimizer, 
                    'lr_scheduler':lr_scheduler }


# Neural Models

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

MAP_NAME_NEURALMODEL = {
    'MLP': MLP
}
