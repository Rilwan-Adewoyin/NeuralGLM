import netCDF4
from netCDF4 import Dataset as nDataset
import os, sys
sys.path.append(os.getcwd())
import yaml

import torch
from torchinfo import summary
import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from dataloaders import Era5EobsDataset, MAP_NAME_DSET

from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser 
from glms import NeuralDGLM
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import pytorch_lightning as pl

import os
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from glms import MAP_NAME_GLM
import glob
import glm_utils
import types
import utils
from torch.utils.data._utils.collate import default_collate

from pytorch_lightning.profiler import AdvancedProfiler
from typing import Dict, Any
import utils

def train( train_args, data_args, glm_args, model_args ):
    # Define the trainer    
    root_dir = train_args.ckpt_dir if train_args.ckpt_dir else ''
    dir_model = os.path.join(root_dir,"Checkpoints",f"{train_args.exp_name}/{train_args.dataset}_{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}")
    
    # Adjusting val_check_interval
    # If val_check_interval is a float then it represents proportion of an epoch
    if train_args.dataset == "uk_rain" and not train_args.val_check_interval.is_integer() :
        batch_count = (data_args.train_set_size_elements // train_args.batch_size )
        if data_args.shuffle:
            # Shuffling of starting date in 7 day period for each location
            # shuffling can drop up to one element per cache_gen_size, per location batch each iteration 
            max_locs_per_gen =    5 #TODO {rilwan.adewoyin} - ensure this is calculate properly, based on the gen_size for making the cache
            # currently we loose 5 or 10 elements per gen original containing  data_args.cache_gen_size
            batch_count -= (  max_locs_per_gen* (data_args.train_set_size_elements// (data_args.cache_gen_size) ) ) //train_args.batch_size
            
            
        train_args.val_check_interval = int( train_args.val_check_interval * batch_count )
             
        print("Validation check every {} steps".format(train_args.val_check_interval))
        
    trainer = pl.Trainer(
                        accelerator='gpu',
                        devices=train_args.devices,
                        default_root_dir = dir_model,
                        callbacks =[EarlyStopping(monitor="val_loss/loss", patience=3 if data_args.locations!=["All"] else 3 ),
                                        ModelCheckpoint(
                                            monitor="val_loss/loss",
                                            filename='{epoch}-{step}-{val_loss/loss:.3f}-{val_metric/mse_rain:.3f}',
                                            save_last=False,
                                            auto_insert_metric_name=True,
                                            save_top_k=1)],
                        enable_checkpointing=True,
                        precision=32,
                        max_epochs=train_args.max_epochs,
                        num_sanity_val_steps=0,
                        limit_train_batches=20 if train_args.debugging else None,
                        limit_val_batches=5 if train_args.debugging else None,
                        limit_test_batches=5 if train_args.debugging else None,
                        val_check_interval=None if train_args.debugging else train_args.val_check_interval
                        )

    # Generate Dataset 
    ds_train, ds_val, ds_test, scaler_features, scaler_target = Era5EobsDataset.get_dataset( data_args,
                                                                    target_distribution_name=glm_args.target_distribution_name,
                                                                    target_range=glm_args.target_range,
                                                                    # gen_size=data_args.gen_size,
                                                                    # cache_gen_size = data_args.cache_gen_size,                             
                                                                    workers = train_args.workers,
                                                                    workers_test = train_args.workers_test,
                                                                    trainer_dir=trainer.logger.log_dir )
    cf = glm_utils.default_collate_concat
    worker_init_fn=Era5EobsDataset.worker_init_fn
            
    # Needs to be true for suffling
    assert ds_train.rain_data.data_len_per_location/ds_train.rain_data.lookback == ds_train.mf_data.data_len_per_location/ds_train.mf_data.lookback, "Per location Length of target and feature do not match {ds_train.rain_data.data_data_len_per_locationlen}-{ds_train.mf_data.data_len_per_location}"


    # Load GLM Model
    glm_class = MAP_NAME_GLM[train_args.glm_name]
    nn_params = { **vars(model_args),
                    **dict( input_shape = data_args.input_shape,
                            output_shape = data_args.output_shape,
                            )}
                    

    nn_params['lookback'] = data_args.lookback_target
    nn_params['tfactor'] = data_args.lookback_feature // data_args.lookback_target

    glm = glm_class(nn_name=train_args.nn_name, 
                        nn_params = nn_params,
                        scaler_target=scaler_target,
                        scaler_features=scaler_features,
                        **vars(glm_args),
                        min_rain_value=data_args.min_rain_value,
                        task = train_args.dataset,
                        debugging=train_args.debugging,
                        dconfig= data_args)    
           
   
    # Create the DataLoaders    
    dl_train =  DataLoader(ds_train, train_args.batch_size,
                            num_workers=train_args.workers,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=cf, 
                            worker_init_fn=worker_init_fn,
                            persistent_workers=False #train_args.workers>0,
                            )

    dl_val = DataLoader(ds_val, train_args.batch_size, 
                            num_workers=train_args.workers,
                            drop_last=False, collate_fn=cf,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            persistent_workers=False #train_args.workers>0,
                            )

    dl_test = DataLoader(ds_test, train_args.batch_size_test, 
                            num_workers=train_args.workers_test,
                            drop_last=False, collate_fn=cf, 
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            prefetch_factor=train_args.prefetch_test,
                            persistent_workers=False ) #train_args.workers>0)
       
    
    # Patching ModelCheckpoint checkpoint name creation
    mc = next( filter( lambda cb: isinstance(cb, ModelCheckpoint), trainer.callbacks) )
    mc._format_checkpoint_name = types.MethodType(utils._format_checkpoint_name, mc)

    if not train_args.test_version:
        # Save the scalers
        glm.save_scalers( trainer.logger.log_dir, dconfig=data_args, glmconfig=glm_args )

        # Fit the Trainer
        try:
            trainer.fit(
                        glm, 
                        train_dataloaders=dl_train,
                        val_dataloaders=dl_val
                         )
        except RuntimeError as e:
            print("e")
            raise e

    # Test the Trainer
    trainer.test(dataloaders=dl_test, ckpt_path='best')

def test( train_args, data_args, glm_args ):

    root_dir = train_args.ckpt_dir if train_args.ckpt_dir else ''
    dir_model = os.path.join(root_dir, f"Checkpoints/{train_args.exp_name}/{train_args.dataset}_{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}")
    dir_model_version = os.path.join(dir_model, "lightning_logs",f"version_{train_args.test_version}")
        
    hparams_path = os.path.join( dir_model_version, "hparams.yaml")
    
    hparams = yaml.load(open(hparams_path,"r"), yaml.UnsafeLoader)
    
    checkpoint_path = next( ( elem for elem in glob.glob(os.path.join( dir_model_version, "checkpoints", "*")) 
                                if elem[-4:]=="ckpt"))
    scaler_features, scaler_target = NeuralDGLM.load_scalers( os.path.join( dir_model_version) )

    #Loading state
    glm_class = MAP_NAME_GLM[train_args.glm_name]
    glm = glm_class.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path, 
                        scaler_features=scaler_features,
                        scaler_target=scaler_target,
                        save_hparams=False)
    
    trainer = pl.Trainer(
                        accelerator='gpu',
                        resume_from_checkpoint=checkpoint_path, 
                        precision=32,
                        enable_checkpointing=not train_args.debugging,
                        logger=False,
                        devices=train_args.devices,
                        default_root_dir = dir_model_version,
                        )

    # Making Dset
    
    saved_dconfig = vars(hparams['dconfig'])
    for k in saved_dconfig:
        if hasattr(data_args, k) and k not in ['test_start','test_end','locations_test','gen_size_test','cache_gen_size_test','test_set_size_elements','loc_count_test']:
            setattr(data_args, k, saved_dconfig[k] )
    
    
    
    ds_test = Era5EobsDataset.get_test_dataset( data_args,
                                                target_distribution_name=glm_args.target_distribution_name,
                                                scaler_features=scaler_features,
                                                scaler_target=scaler_target,
                                                gen_size=data_args.gen_size_test,
                                                workers = train_args.workers_test ) 

    dl_test = DataLoader(ds_test, train_args.batch_size_test, 
                            shuffle=False,
                            num_workers=train_args.workers_test,
                            prefetch_factor=train_args.prefetch_test,
                            drop_last=False, collate_fn=glm_utils.default_collate_concat, 
                            worker_init_fn=Era5EobsDataset.worker_init_fn)

    # Test the Trainer
    trainer.test(glm, dataloaders=dl_test)


def parse_train_args(parent_parser=None, list_args=None):
    

    # Train args
    train_parser = argparse.ArgumentParser(parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)
    train_parser.add_argument("--exp_name", default='default', type=str )        
    train_parser.add_argument("--devices", default=1)
    train_parser.add_argument("--sample_size", default=100)
    train_parser.add_argument("--dataset", default="uk_rain", choices=["toy","australia_rain","uk_rain"])
    train_parser.add_argument("--nn_name", default="HConvLSTM_tdscale", choices=["MLP", "HLSTM", "HLSTM_tdscale", "HConvLSTM_tdscale"])
    train_parser.add_argument("--glm_name", default="DGLM", choices=["DGLM"])
    train_parser.add_argument("--max_epochs", default=300, type=int)
    train_parser.add_argument("--batch_size", default=24, type=int)
    train_parser.add_argument("--batch_size_test", default=720, type=int)
    
    train_parser.add_argument("--debugging",action='store_true', default=False )
    train_parser.add_argument("--workers", default=8, type=int )
    train_parser.add_argument("--workers_test", default=8, type=int )
       
    
    train_parser.add_argument("--test_version",default=None, type=int, required=False ) 
    train_parser.add_argument("--val_check_interval", default=1.0, type=float)
    train_parser.add_argument("--prefetch",type=int, default=2, help="Number of batches to prefetch" )
    train_parser.add_argument("--prefetch_test",type=int, default=4, help="Number of batches to prefetch" )
    train_parser.add_argument("--hypertune",type=bool, default=False)
    train_parser.add_argument("--ckpt_dir",type=str, default='')
    
    
    train_args = train_parser.parse_known_args(args=list_args)[0]

    return train_args

if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # # Train args
    # train_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
    # train_parser.add_argument("--exp_name", default='default', type=str )        
    # train_parser.add_argument("--devices", default=1)
    # train_parser.add_argument("--sample_size", default=100)
    # train_parser.add_argument("--dataset", default="uk_rain", choices=["toy","australia_rain","uk_rain"])
    # train_parser.add_argument("--nn_name", default="HConvLSTM_tdscale", choices=["MLP","HLSTM","HLSTM_tdscale", "HConvLSTM_tdscale"])
    # train_parser.add_argument("--glm_name", default="DGLM", choices=["DGLM"])
    # train_parser.add_argument("--max_epochs", default=300, type=int)
    # train_parser.add_argument("--batch_size", default=24, type=int)
    # train_parser.add_argument("--batch_size_test", default=720, type=int)
    
    # train_parser.add_argument("--debugging",action='store_true', default=False )
    # train_parser.add_argument("--workers", default=2, type=int )
    # train_parser.add_argument("--workers_test", default=6, type=int )
       
    
    # train_parser.add_argument("--test_version",default=None, type=int, required=False ) 
    # train_parser.add_argument("--val_check_interval", default=1.0, type=float)
    # train_parser.add_argument("--prefetch",type=int, default=2, help="Number of batches to prefetch" )
    # train_parser.add_argument("--prefetch_test",type=int, default=4, help="Number of batches to prefetch" )
    # train_parser.add_argument("--hypertune",type=bool, default=False)
    # train_parser.add_argument("--ckpt_dir",type=str, default='')
    
    
    # train_args = train_parser.parse_known_args()[0]
    train_args = parse_train_args(parent_parser)
    
    # add model specific args
    model_args = MAP_NAME_NEURALMODEL[train_args.nn_name].parse_model_args(parent_parser)
    
    # add data specific args
    data_args = MAP_NAME_DSET[train_args.dataset].parse_data_args(parent_parser)

    # add glm specific args
    glm_args = MAP_NAME_GLM[train_args.glm_name].parse_glm_args(parent_parser)
    
    if train_args.test_version==None:
        train(train_args, data_args, glm_args, model_args)
    else:
        test( train_args, data_args, glm_args )
    

    




