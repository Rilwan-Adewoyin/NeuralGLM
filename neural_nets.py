from typing import Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
from loss_utils import LossMixin
from better_lstm import LSTM
import einops
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import torchtyping
import argparse
from torch.nn import Parameter
#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git


class MLP(nn.Module):

    model_type = "MLP"

    def __init__(self, input_shape=(6,), output_shape=(1,), dropout=0.0 ,**kwargs ):
        super().__init__()

        raise NotImplementedError
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropout = dropout
        
        self.encoder = nn.Sequential(nn.Linear( *self.input_shape, 32), 
                                nn.SELU() ,
                                nn.Linear(32, 16),
                                nn.SELU() ) 

        self.outp_mean = nn.Linear(16,*output_shape)
        self.outp_dispersion =  nn.Linear(16, *self.output_shape)

    def forward(self, x):
        h = self.encoder(x)
        pred_mean = self.outp_mean(h)
        pred_dispersion = self.outp_dispersion(h)
        return pred_mean, pred_dispersion

    @staticmethod
    def parse_model_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        model_args = parser.parse_known_args()[0]
        return model_args

class HLSTM(nn.Module):
    
    model_type = "HLSTM"

    def __init__(self,
                    input_shape=(6,),
                    output_shape=(2,),
                    hidden_dim:int=32,
                    num_layers:int=2, 
                    hurdle_model:bool=False,
                    zero_inflated_model:bool=False ) -> None:
        """[summary]

        Args:
            input_shape (tuple, optional): [description]. Defaults to (6,).
            output_shape (tuple, optional): [description]. Defaults to (2,).
            hidden_dim (int, optional): [Dimensions of hidden layers in model]. Defaults to 32.
            num_layers (int, optional): [Number of layers in neural network]. Defaults to 2.
            hurdle_model (bool, optional): [Whether or not we use a hurdle type model]. Defaults to False.
            zero_inflated_model (bool, optional): [Whether or not we use a zero inflated model]. Defaults to False.
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hurdle_model = hurdle_model
        self.zero_inflated_model = zero_inflated_model

        self.upscale = nn.Sequential( nn.Linear( input_shape[0], hidden_dim, bias=False ), nn.SELU() )


        self.encoder = nn.Sequential(
            LSTM( input_size = hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                    dropouti=0.20, dropoutw=0.35, dropouto=0.25,
                    batch_first=True, bidirectional=True),
            ExtractLSTMOutputFeatures()
        )

        if hurdle_model:
            self.outp_logitsrain = nn.Sequential(  nn.Linear(hidden_dim*2, hidden_dim, bias=False), nn.SELU(), nn.Linear(hidden_dim, *self.output_shape, bias=False) )

        self.outp_mean = nn.Sequential( nn.Linear(hidden_dim*2, *self.output_shape, bias=False),  nn.ReLU())
        self.outp_dispersion = nn.Sequential( nn.Linear(hidden_dim*2, *self.output_shape, bias=False), nn.ReLU())
        
    def forward(self, x, standardized_output=True):
        x = self.upscale(x)
        h = self.encoder(x)

        output = {}

        if self.hurdle_model:
            output['logits'] = self.outp_logitsrain(h)

        output['mean'] = self.outp_mean(h)
        output['disp'] = self.outp_dispersion(h)
        
        return output
    
    @staticmethod
    def parse_model_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        model_args = parser.parse_known_args()[0]
        return model_args

class ExtractLSTMOutputFeatures(nn.Module):
    """
        Module that extracts the hidden state output from an LSTM based layer in torch
    """
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
    
MAP_NAME_NEURALMODEL = {
    'MLP': MLP,
    'HLSTM': HLSTM
}
