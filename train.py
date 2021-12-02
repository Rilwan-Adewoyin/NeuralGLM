import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
from models import NeuralDGLM
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from dataloaders import ToyDataset
from models import MAP_NAME_NEURALMODEL
import pickle
import json


if __name__ == '__main__':

    #TODO: add arugments so it can be run
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--sample_size", default=1000)
    parser.add_argument("--input_shape", default=(6,) )
    parser.add_argument("--output_shape", default=(1,) )
    parser.add_argument("--model_name", default="MLP")
    parser.add_argument("--target_params", default=[0, 3, 0.5], type=lambda _str: json.loads(_str) )

    args = parser.parse_args()

    # Load Datasets
    target_func_coeffs = { 'c':0, 'x':torch.randint(0,3,args.input_shape), 'x^2':torch.randint(0,3,args.input_shape) }
    tfc = target_func_coeffs
    
    # Sampling Dataset
    loc = torch.zeros(args.input_shape)
    scale = torch.ones(args.input_shape)
    ds_train, ds_val, ds_test = ToyDataset.get_toy_dataset(
            input_distribution='mv_lognormal',
            inp_sample_params={ 'loc':loc, 'scale':scale },
            
            sample_size=args.sample_size,

            target_func= lambda inp: torch.sum( tfc['c'] + inp*tfc['x'] + inp*tfc['x^2'], dim=-1 ),

            noise_method = 'random_guassian',
            noise_sample_params= { 'loc':0, 'scale':0.1 }

            )

    # Create the DataGenerators
    dl_train =  DataLoader(ds_train, 50, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_train, 50, shuffle=False, num_workers=2)
    dl_test = DataLoader(ds_train, 50, shuffle=False, num_workers=2)
    
    # Loading BaseModel
    model = MAP_NAME_NEURALMODEL['MLP']( args.input_shape, args.output_shape)

    # Create DGLM Structure
    glm = NeuralDGLM(
        model=model,
        model_params = None,
        target_distribution_name='lognormal', 
        
        mean_distribution_name='normal',
        mean_link_name='identity', 

        dispersion_distribution_name='gamma', 
        dispersion_link_name='relu_inverse' )

    # Define the trainer
    trainer = pl.Trainer(   gpus=args.gpus,
                            default_root_dir = f"toy_experiments_logs/{glm.glm_type}_{glm.model.model_type}",
                            callbacks =[EarlyStopping(monitor="val_loss"),
                                            ModelCheckpoint(
                                                monitor="val_loss",
                                                save_last=True,
                                                auto_insert_metric_name=True,
                                                save_top_k=2)
                                             ] ,
                            checkpoint_callback=True,
                            max_epochs=30 )

    # Fit the Trainer
    trainer.fit(glm, 
                    train_dataloaders=dl_train,
                    val_dataloaders=dl_val )

    trainer.test(test_dataloaders=dl_test)


