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
from dataloaders import generate_dataset
from models import MAP_NAME_NEURALMODEL
import pickle
import json

if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # data_parser = ArgumentParser(
    #         parents=[parent_parser], add_help=True, allow_abbrev=False)

    # model_parser = ArgumentParser(
    #         parents=[parent_parser], add_help=True, allow_abbrev=False)
    
    # training_parser = ArgumentParser(
    #         parents=[parent_parser], add_help=True, allow_abbrev=False)

    parser = ArgumentParser()

    parser.add_argument("--gpus", default=None)
    parser.add_argument("--sample_size", default=1000)
    parser.add_argument("--dataset", default="australia_rain", choices=["toy","australia_rain"])
    
    #Model args
    parser.add_argument("--input_shape", type=tuple, default=(19,) )
    parser.add_argument("--output_shape", default=(1,) )

    #GLM args
    parser.add_argument("--target_distribution_name", default="lognormal_hurdle" )
        
    #ToyDataset args 
    parser.add_argument("--target_params", default=[0, 3, 0.5], type=lambda _str: json.loads(_str) )

    #AustraliaDataset
    parser.add_argument("--lookback", default=7, type=int )
    parser.add_argument("--model_name", default="HLSTM", choices=["MLP","HLSTM"])

    args = parser.parse_args()    
    
    _ = vars(args)
    ds_train, ds_val, ds_test, sclaer_features, scaler_targets = generate_dataset( args.dataset, **_ )

    # Create the DataGenerators
    dl_train =  DataLoader(ds_train, 10, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_train, 10, shuffle=False, num_workers=2)
    dl_test = DataLoader(ds_train, 10, shuffle=False, num_workers=2)
    
    # Loading BaseModel
    model = MAP_NAME_NEURALMODEL[args.model_name]( args.input_shape, args.output_shape,
        hurdle_model = "hurdle" in args.target_distribution_name )

    # Create DGLM Structure
    #TODO: need to ensure that sampling is zero minimized
    glm = NeuralDGLM(
        model=model,
        model_params = None,

        target_distribution_name='lognormal_hurdle', 
        
        mean_distribution_name='normal',
        mean_link_name='identity', 
        mean_link_func_params={},

        dispersion_distribution_name='gamma', 
        dispersion_link_name='relu_inverse',

        scaler_targets=scaler_targets,
        scaler_features=sclaer_features )

    # Define the trainer
    trainer = pl.Trainer(   gpus=args.gpus,
                            default_root_dir = f"Checkpoints/{args.dataset}/{glm.glm_type}_{glm.model.model_type}",
                            callbacks =[EarlyStopping(monitor="val_loss", patience=3),
                                            ModelCheckpoint(
                                                monitor="val_loss",
                                                filename='{epoch}-{step}-{val_loss:.4f}',
                                                save_last=True,
                                                auto_insert_metric_name=True,
                                                save_top_k=2)
                                             ] ,
                            checkpoint_callback=True,
                            precision=32,
                            max_epochs=30 )

    # Fit the Trainer
    trainer.fit(glm, 
                    train_dataloaders=dl_train,
                    val_dataloaders=dl_val )

    trainer.test(ckpt_path='best', test_dataloaders=dl_test)


