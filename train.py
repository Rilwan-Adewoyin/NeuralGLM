import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from argparse import ArgumentParser
from glms import NeuralDGLM
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pickle
import pytorch_lightning as pl
import json
from dataloaders import ToyDataset, AustraliaRainDataset, MAP_NAME_DSET
import torchtyping
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from glms import MAP_NAME_GLM


if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # Train args
    train_parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
    train_parser.add_argument("--gpus", default=1)
    train_parser.add_argument("--sample_size", default=1000)
    train_parser.add_argument("--dataset", default="australia_rain", choices=["toy","australia_rain"])
    train_parser.add_argument("--model_name", default="HLSTM", choices=["MLP","HLSTM"])
    train_parser.add_argument("--glm_name", default="DGLM", choices=["DGLM"])

    train_args = train_parser.parse_known_args()[0]
    
    # add model specific args
    model_args = MAP_NAME_NEURALMODEL[train_args.model_name].parse_model_args(parent_parser)
    
    # add data specific args
    data_args = MAP_NAME_DSET[train_args.dataset].parse_data_args(parent_parser)

    # add glm specific args
    glm_args = MAP_NAME_GLM[train_args.glm_name].parse_glm_args(parent_parser)
        
    # Generate Dataset 
    if train_args.dataset == 'toy':

        # Randomly Sample the co-effecients a,b,c
        coeffs = { 'c':torch.rand(data_args.input_shape), 'x':torch.randint(0,3, data_args.input_shape), 'x^2':torch.randint(0,3,data_args.input_shape) }
        target_func = lambda inp: torch.sum( coeffs['c'] + inp*coeffs['x'] + inp*coeffs['x^2'], dim=-1 )
        
        ds_train, ds_val, ds_test, scaler_features, scaler_targets = ToyDataset.get_dataset( target_func=target_func, **vars(data_args))
    
    elif train_args.dataset == "australia_rain":
        ds_train, ds_val, ds_test, scaler_features, scaler_targets = AustraliaRainDataset.get_dataset(**vars(data_args))
        data_args.input_shape = ( len( ds_train.datasets[0].features.columns ), )

    else:
        raise NotImplementedError

    # Create the DataLoaders
    dl_train =  DataLoader(ds_train, 240, shuffle=True, num_workers=6, drop_last=False)
    dl_val = DataLoader(ds_train, 1280, shuffle=False, num_workers=4, drop_last=False)
    dl_test = DataLoader(ds_train, 1280, shuffle=False, num_workers=4, drop_last=False)
    
    # Load Neural Model to be used
    neural_net_class = MAP_NAME_NEURALMODEL[train_args.model_name]
    neural_net = neural_net_class( input_shape = data_args.input_shape,
                            output_shape = data_args.output_shape,
                            hurdle_model = "hurdle" in glm_args.target_distribution_name, 
                            zero_inflated_model = "zero_inflated" in glm_args.target_distribution_name,
                            **vars(model_args) )

    # Load GLM Model
    glm_class = MAP_NAME_GLM[train_args.glm_name]
    glm = glm_class(neural_net=neural_net, 
                        scaler_targets=scaler_targets,
                        scaler_features=scaler_features,
                        **vars(glm_args))

    # Define the trainer
    trainer = pl.Trainer(   gpus=train_args.gpus,
                            default_root_dir = f"Checkpoints/{train_args.dataset}/{train_args.glm_name}_{train_args.model_name}",
                            callbacks =[EarlyStopping(monitor="val_loss", patience=3),
                                            ModelCheckpoint(
                                                monitor="val_loss",
                                                filename='{epoch}-{step}-{val_loss:.4f}',
                                                save_last=True,
                                                auto_insert_metric_name=True,
                                                save_top_k=2)
                                             ] ,
                            enable_checkpointing=True,
                            precision=16,
                            max_epochs=100,
                            log_every_n_steps=10
                             )

    # Fit the Trainer
    trainer.fit(glm, 
                    train_dataloaders=dl_train,
                    val_dataloaders=dl_val )
    
    # Test the Trainer
    trainer.test(dataloaders=dl_test, ckpt_path='best')


