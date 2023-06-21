
import os
from dataloaders import Era5EobsTopoDataset_v2
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Union, Optional
from pathlib import Path
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pickle
import argparse
import yaml
from torch.optim import Adam
from argparse import ArgumentParser
from loss_utils import VAEGANLoss
import glob
from torch.nn import functional as F
from torch.utils.data import DataLoader
import types
import utils
import einops
import torchmetrics as tm
import copy


class GenerativeLightningModule(pl.LightningModule):
    
    def __init__(self, 
            scaler_features:Union[MinMaxScaler,StandardScaler],
            scaler_target:Union[MinMaxScaler,StandardScaler],
            neural_net,            
            debugging=False,
            **kwargs: Any) -> None:
        
        super().__init__()
                
        # Trainer
        self.dconfig = kwargs.get('dconfig', None)
        self.batch_size = kwargs.get('batch_size',None)
        self.sample_size = kwargs.get('sample_size',None)
        
        # Load Neural Model

        self.neural_net = neural_net
        
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        # self.register_buffer('target_scale', torch.as_tensor( scaler_target.scale_[0]) )
        self.target_unscale = lambda x: torch.pow(10,x)-1
        
        self.debugging = debugging
        
        self.vaegan_loss = VAEGANLoss( content_loss=False, 
                                # content_loss_name='ensmeanMSE_phys',
                                kl_weight=1e-5, 
                                gp_weight=10,
                                # cl_weight=0.2  
                                )
        self.mse = tm.MeanSquaredError(squared=True).to('cuda')        
        self.mse_mean = tm.MeanMetric()
        
        self.log_dir = kwargs.get('log_dir',None)
            
    
    def forward(self, variable_fields, constant_fields, mask  ):
    
        score_pred , image_output_scaled, z_mean, z_logvar = self.neural_net( variable_fields, constant_fields, mask  )
        return score_pred , image_output_scaled, z_mean, z_logvar
        
    def training_step(self, batch, batch_idx, optimizer_idx=None ):    
    
        # Extracting data from batch
        variable_fields = batch['fields'].to(self.dtype)
        constant_fields = batch['topo'].to(self.dtype)
        image_scaled = batch['rain'].to(self.dtype)
            # Images set to (c,h,w) == 0.0 correspond to inputs that arent real
            # False datums will have corresponding images of all 0
        variable_fields = einops.rearrange( variable_fields, 'b h w c ->b c h w')
        constant_fields = constant_fields[:, None]  
        image_scaled = image_scaled[:,None]
        
        mask = batch['mask']
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        mask = ~mask    
        
        output = {}
                    
        # if True: 
        if optimizer_idx == 1:
        # generator step

            score_pred , image_output_scaled, z_mean, z_logvar = self.neural_net( variable_fields, constant_fields, mask  )
            
            fake_img_score = self.vaegan_loss.wasserstein( torch.ones_like(score_pred), score_pred )
            
            kl_loss = self.vaegan_loss.kl_loss(z_logvar, z_mean, score_pred.shape[0], mask)
                                    
            output['loss'] = -fake_img_score + kl_loss
                                   
            mse_loss = self.mse(
                self.target_unscale( torch.masked_select(image_scaled.squeeze(1), mask) ),
                    self.target_unscale( torch.masked_select(image_output_scaled.squeeze(1), mask) ) 
                )

            self.log("loss/gen", output['loss'], on_step=False, on_epoch=True, prog_bar=False)               
            self.log("loss/kl", kl_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("loss/fake_img_score", fake_img_score, on_step=False, on_epoch=True, prog_bar=False)
                                
        elif optimizer_idx == 0:
            # Discrimintor step
                    
            with torch.no_grad():
                vf = einops.rearrange( variable_fields, '(b t) d h w -> b (t d) h w', t=4)
                vf = self.neural_net.grouped_conv2d_reduce_time( vf )
            
            # Real images score
            score_pred  = self.neural_net.discriminator( image_scaled, vf, constant_fields, mask  )
            real_img_score = self.vaegan_loss.wasserstein( torch.ones_like(score_pred), score_pred)
            
            # Fake image score
            score_pred_fake, image_output_scaled, z_mean, z_logvar = self.forward( variable_fields, constant_fields, mask  )
            fake_img_score = self.vaegan_loss.wasserstein( torch.ones_like(score_pred), score_pred_fake) 
            
            # Gradient Penalty
            gp_loss = self.vaegan_loss.gradient_penalty(
                            self.neural_net.discriminator, 
                            image_scaled, 
                            image_output_scaled,
                            disc_args =  (vf, constant_fields, mask))
                        
            loss_disc = -real_img_score + fake_img_score + gp_loss
            self.log("loss/disc", loss_disc, on_step=False, on_epoch=True, prog_bar=False)
            output = {'loss':loss_disc}
            
            self.log("loss/real_img_score", real_img_score, on_step=False, on_epoch=True, prog_bar=False)
            self.log("loss/fake_img_score", fake_img_score, on_step=False, on_epoch=True, prog_bar=False)
            self.log("loss/gp_loss", gp_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("loss", output['loss'])
            
            mse_loss = self.mse(
                self.target_unscale( torch.masked_select(image_scaled.squeeze(1), mask) ),
                    self.target_unscale( torch.masked_select(image_output_scaled.squeeze(1), mask) ) 
                )
        
        self.log("loss/mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True) 
            
        return output

   
    
    def validation_step(self, batch, batch_idx):
                
        # Extracting data from batch
        variable_fields = batch['fields'].to(self.dtype)
        constant_fields = batch['topo'].to(self.dtype)
        image_scaled = batch['rain'].to(self.dtype) # Images set to (c,h,w) == 0.0 correspond to inputs that arent real
                                # False datums will have corresponding images of all 0
        mask = batch['mask']
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        mask = ~mask    
        
        variable_fields = einops.rearrange( variable_fields, 'b h w c ->b c h w',)
        constant_fields = constant_fields[:, None]
        
        image_output_scaled, *_= self.neural_net.generator( variable_fields, constant_fields )
        

        
        mse_score =  self.mse(
                        self.target_unscale( torch.masked_select(image_scaled, mask)),
                        self.target_unscale( torch.masked_select(image_output_scaled.squeeze(1), mask))
                        )
        
        self.mse_mean.update(mse_score)
        
        return None
    
    def validation_epoch_end(self, validation_step_outputs):
        output = {}
        
        
        mse_score = self.mse_mean.compute()

        self.log("val_mse", mse_score, prog_bar=True, on_epoch=True)
        
        self.mse_mean.reset()
        
    def test_step(self, batch, batch_idx) :
        output = {}
               
        # Extracting data from batch
        
        variable_fields = batch['fields'].to(self.dtype)
        constant_fields = batch['topo'].to(self.dtype)
        
        image_scaled = batch['rain'].to(self.dtype) 
                    # Images set to (c,h,w) == 0.0 correspond to inputs that arent real
                    # False datums will have corresponding images of all 0
        mask = batch['mask']

        variable_fields = einops.rearrange( variable_fields, 'b h w c ->b c h w')
        constant_fields = constant_fields[:, None]  
            
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        mask = ~mask        
        target_date_window = batch.pop('target_date_window', None)
        
        # generating image
        if self.sample_size is None:
            image_output_scaled_mean, *_= self.neural_net.generator( variable_fields, constant_fields )
        else:
            li_image_output_scaled = [None]*self.sample_size
            
            for idx in range(self.sample_size):
                i_o_s, *_ = self.neural_net.generator( variable_fields, constant_fields )
                li_image_output_scaled[idx]=i_o_s
            image_output_scaled_ensemble = torch.stack( li_image_output_scaled, -1 )
            image_output_scaled_mean = image_output_scaled_ensemble.mean(-1)
                           
        # Scaling up image
        output = {}
        output['pred_rain_mean'] = (self.target_unscale( image_output_scaled_mean).squeeze().cpu().numpy() )
        if self.sample_size is not None:
            output['pred_rain_ensemble'] = (self.target_unscale( image_output_scaled_ensemble).squeeze().cpu().numpy() )
        output['target_rain'] = (self.target_unscale(image_scaled).cpu().numpy() )
        output['mask'] = mask.cpu().numpy()
        output['target_date_window'] = target_date_window
        return output
    
    def test_epoch_end(self, outputs) -> None:
        print("\nSaving Test Output to File")

        # Concatenating outputs of test steps
        pred_rain_mean = np.concatenate( [d['pred_rain_mean'] for d in outputs] )
        target_rain = np.concatenate( [d['target_rain'] for d in outputs] )
        mask = np.concatenate([d['mask'] for d in outputs])
        target_date_window = np.concatenate( [d['target_date_window'] for d in outputs] )

        # sorting outputs based on dates
        sort_idx = np.argsort(target_date_window)
        pred_rain_mean = pred_rain_mean[sort_idx]
        target_rain = target_rain[sort_idx]
        mask = mask[sort_idx]
        target_date_window = target_date_window[sort_idx]
        
                
        # Saving Model Outputs To file 
        output = {}
        output['pred_rain_mean'] = pred_rain_mean
        output['target_rain'] = target_rain
        output['mask'] = mask
        output['target_date_window'] = target_date_window
        
        # Adding ensemble info to output dict
        log_dir = self.log_dir if self.log_dir else self.logger.log_dir
        
        if self.sample_size is not None:
            pred_rain_ensemble = np.concatenate( [d['pred_rain_ensemble'] for d in outputs] )
            pred_rain_ensemble = pred_rain_ensemble[sort_idx]
            output['pred_rain_ensemble']  = pred_rain_ensemble
            
        suffix = f"{self.dconfig.test_start}_{self.dconfig.test_end}"
        file_path = os.path.join( log_dir , f"test_output_{suffix}.pkl" ) 
        with open( file_path, "wb") as f:
            pickle.dump( output, f )
            
        # Logging Model Evaluation Metrics
        
        sqd_diff = (target_rain-pred_rain_mean)**2
        mse_score = sqd_diff[mask].mean()
        r10mse_score = sqd_diff[ np.logical_and(mask, target_rain>=10.0 )  ].mean()
                
        self.log("test_mse", mse_score, prog_bar=True, on_epoch=True)
        self.log("test_r10mse", r10mse_score, prog_bar=True, on_epoch=True)
        
        # Saving Model Evaluation Metrics
        # Recording losses on test set and summarised information about test run
        summary = {
            'train_start': self.dconfig.train_start,
            'train_end': self.dconfig.train_end,
            'val_start': self.dconfig.val_start,
            'val_end': self.dconfig.val_end,
            'test_start': self.dconfig.test_start,
            'test_end': self.dconfig.test_end,
            'test_mse': mse_score.item(),
            
            'test_r10mse': r10mse_score.item()}

        file_path_summary = os.path.join(log_dir, f"summary_{suffix}.yaml")
        with open(file_path_summary, "w") as f:
            yaml.dump( summary, f)
        return super().test_epoch_end(outputs)
    
    def configure_optimizers(self):
        
        
        generator_params = (*self.neural_net.encoder.parameters(), 
                            *self.neural_net.grouped_conv2d_reduce_time.parameters(),
                            *self.neural_net.decoder.parameters() )
        
        lr_scale = self.batch_size/32
        
        lr = (5e-6)*lr_scale
                
        optimizer_generator = Adam(  generator_params , lr=lr, betas=(0.5,0.9))
        frequency_generator = 1 
                
        dict_generator = {
            'optimizer': optimizer_generator,   
            'frequency': frequency_generator
        }
        
        discriminator_params = self.neural_net.discriminator.parameters()
        optimizer_discriminator = Adam(discriminator_params , lr=lr, betas=(0.5,0.9) )
        frequency_discriminator = 5
        dict_discriminator = {
            'optimizer': optimizer_discriminator,
            'frequency': frequency_discriminator,
        }
                
        return (dict_discriminator, dict_generator)
        
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_save_checkpoint(checkpoint)
        try:
            mc = next( filter( lambda cb: isinstance(cb, ModelCheckpoint), self.trainer.callbacks) )
            if not os.path.exists(mc.dirpath): os.makedirs(mc.dirpath, exist_ok=True)
            mc.to_yaml()
        except FileNotFoundError as e:
            pass

        return res        

    def load_configs( dir_path:str ):
         
        path_config_train = os.path.join(dir_path, "config_trainer.yaml")
        path_config_data = os.path.join(dir_path, "config_data.yaml")
        path_config_model = os.path.join(dir_path, "config_model.yaml")
        
        config_train = yaml.unsafe_load(open(path_config_train,"r"))
        config_data = yaml.unsafe_load(open(path_config_data,"r"))
        config_model = yaml.unsafe_load(open(path_config_model,"r"))
        
        return config_train, config_data, config_model
    
    @staticmethod
    def load_scalers(dir_path:str):
        """Loads the scaler object from a given path.
                The path can be a path to the directory containing both the feature and target scaler

        Args:
            dir_path (str, optional): [description]. Defaults to None.


        Returns:
            [type]: [description]
        """

        path_scaler_features = os.path.join(dir_path, "scaler_features.pkl")
        path_scaler_target = os.path.join(dir_path, "scaler_target.pkl")
        path_scaler_topo = os.path.join(dir_path, "scaler_topo.pkl")

        if os.path.exists(path_scaler_features) and os.path.exists(path_scaler_target):
            scaler_features = pickle.load(open(path_scaler_features,"rb"))
            scaler_target = pickle.load(open(path_scaler_target,"rb"))
            scaler_topo = pickle.load(open(path_scaler_topo,"rb"))
        else:
            raise FileNotFoundError(f"The feature and target scalers can not be found at the directory below:\n {dir_path}") 
        
        return scaler_features, scaler_target, scaler_topo

    def save_configs(self, train_args, data_args, model_args, dir_ ):
        
        os.makedirs(dir_, exist_ok=True)
        
        yaml.dump(train_args, open( os.path.join(dir_, "config_trainer.yaml"), "w" ) )
        yaml.dump(data_args, open( os.path.join(dir_, "config_data.yaml"), "w" ) )
        yaml.dump(model_args, open( os.path.join(dir_, "config_model.yaml"), "w" ) )
        
    def save_scalers(self, scaler_features, scaler_target, scaler_topo, dir_) -> None:
        
        os.makedirs(dir_, exist_ok=True)
        
        pickle.dump(scaler_features, open(os.path.join(dir_, "scaler_features.pkl"),"wb") )
        pickle.dump(scaler_target, open(os.path.join(dir_, "scaler_target.pkl"),"wb") )
        pickle.dump(scaler_topo, open(os.path.join(dir_, "scaler_topo.pkl"),"wb") )

    @staticmethod
    def parser_trainer_args(parent_parser):
        
        # Train args
        train_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        train_parser.add_argument("--exp_name", default='vaegan_benchmark', type=str )        
        train_parser.add_argument("--devices", default=1, type=int)
        
        train_parser.add_argument("--sample_size", default=500, type=int)
                
        train_parser.add_argument("--max_epochs", default=300, type=int)
        train_parser.add_argument("--batch_size", default=48, type=int)
        train_parser.add_argument("--batch_size_inf", default=128, type=int)
                
        train_parser.add_argument("--debugging",action='store_true', default=False )
        train_parser.add_argument("--workers", default=6, type=int )
                        
        train_parser.add_argument("--test_version", default=None, type=int, required=False ) 
        train_parser.add_argument("--val_check_interval", default=1.0, type=float)
        train_parser.add_argument("--prefetch", type=int, default=2, help="Number of batches to prefetch" )
                
        train_parser.add_argument("--ckpt_dir", type=str, default='./VAEGAN/Checkpoints')
        
        train_args = train_parser.parse_known_args()[0]
                
        return train_args

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

    @staticmethod
    def train_model( train_args, data_args, model_args ):
        # Define the trainer    
        root_dir = train_args.ckpt_dir if train_args.ckpt_dir else ''
        dir_model = os.path.join(root_dir,f"{train_args.exp_name}")
        
        # Adjusting val_check_interval
        # If val_check_interval is a float then it represents proportion of an epoc
        trainer = pl.Trainer(devices=train_args.devices,
                            default_root_dir = dir_model,
                            callbacks = [EarlyStopping(monitor="val_mse", patience=1000),
                                            ModelCheckpoint(
                                                monitor="val_mse",
                                                filename='{epoch}-{step}-{val_mse:.3f}',
                                                save_last=False,
                                                auto_insert_metric_name=True,
                                                save_top_k=1)],
                            enable_checkpointing=True,
                            precision=32,
                            max_epochs=train_args.max_epochs,
                            num_sanity_val_steps=0,                            
                            limit_train_batches=51 if train_args.debugging else None,
                            limit_val_batches=51 if train_args.debugging else None,
                            limit_test_batches=51 if train_args.debugging else None,
                            val_check_interval= train_args.val_check_interval
                            )
        # Load GLM Model
        neural_net = VAEGAN(**vars(model_args) )
                
        # Generate Dataset
        ds_train, ds_val, ds_test, scaler_features, scaler_target, scaler_topo = Era5EobsTopoDataset_v2.get_datasets( data_args)
                               
   
        # Create the DataLoaders    
        dl_train =  DataLoader(ds_train, train_args.batch_size,
                                num_workers=train_args.workers,
                                drop_last=True,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn, 
                                persistent_workers=True)

        dl_val = DataLoader(ds_val, train_args.batch_size_inf, 
                                num_workers=train_args.workers,
                                drop_last=False, 
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn,
                                pin_memory=True,
                                persistent_workers=True)

        dl_test = DataLoader(ds_test, train_args.batch_size_inf, 
                                drop_last=False,
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn,
                                pin_memory=True,
                                persistent_workers=False )
        

        # Define Lightning Module
        glm = GenerativeLightningModule(scaler_features,scaler_target,
                                        neural_net,
                                        train_args.debugging,
                                        dconfig = data_args,
                                        sample_size = train_args.sample_size,
                                        batch_size = train_args.batch_size)

        # Patching ModelCheckpoint checkpoint name creation
        mc = next( filter( lambda cb: isinstance(cb, ModelCheckpoint), trainer.callbacks) )
        mc._format_checkpoint_name = types.MethodType(utils._format_checkpoint_name, mc)

    
        # Save the config scalers
        # dir_ = os.path.join(trainer.logger._save_dir, 'lightning_logs' ,str(trainer.logger._version))
        glm.save_configs( train_args, data_args, model_args, trainer.logger.log_dir )
        glm.save_scalers( scaler_features, scaler_target, scaler_topo, trainer.logger.log_dir )
            
        # Fit the Trainer
        trainer.fit(  glm, 
                        train_dataloaders=dl_train,
                        val_dataloaders=dl_val if not train_args.debugging else copy.deepcopy(dl_train))
        
        # Test the Trainer
        trainer.test(dataloaders=dl_test, ckpt_path='best')

    @staticmethod
    def test_model( train_args, data_args):

        root_dir = train_args.ckpt_dir if train_args.ckpt_dir else ''
        dir_model = os.path.join(root_dir,f"{train_args.exp_name}")
        dir_model_version = os.path.join(dir_model, "lightning_logs",f"version_{train_args.test_version}")
        
        # Load Confifs and Scalers
        # allowing user to update test parameters used
        changed_args_t = { k:getattr(train_args,k) for k in ['sample_size','batch_size_inf'] if hasattr(train_args,k) }
        changed_args_d = { k:getattr(data_args,k) for k in ['test_start','test_end','data_dir'] if hasattr(data_args,k) }
        train_args, data_args, model_args = GenerativeLightningModule.load_configs( dir_model_version )
        
        for k,v in changed_args_t.items(): setattr(train_args,k,v)
        for k,v in changed_args_d.items(): setattr(data_args,k,v)
        
        scaler_features, scaler_target, scaler_topo = GenerativeLightningModule.load_scalers( dir_model_version )
        
        checkpoint_path = next( ( elem for elem in glob.glob(os.path.join( dir_model_version, "checkpoints", "*")) 
                                    if elem[-4:]=="ckpt"))
        
        neural_net=VAEGAN(**vars(model_args))
        
        trainer = pl.Trainer(  
                    precision=16,
                    enable_checkpointing=True,
                    logger=False,
                    devices=train_args.devices,
                    default_root_dir=dir_model_version)

        # Making Dset
        ds_test = Era5EobsTopoDataset_v2( start_date=data_args.test_start, end_date=data_args.test_end,
                                         dconfig=data_args,
                                    xarray_decode=True,
                                    scaler_features=scaler_features, 
                                    scaler_target=scaler_target,
                                    scaler_topo=scaler_topo)
    
        dl_test = DataLoader(ds_test, train_args.batch_size, 
                                drop_last=False,
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn,
                                pin_memory=True,
                                persistent_workers=False )

        # Define Lightning Module
        lightning_module = GenerativeLightningModule(scaler_features,scaler_target, neural_net, log_dir=dir_model_version, dconfig=data_args)
      
        
        # Test the Trainer
        trainer.test(
            lightning_module,
            ckpt_path=checkpoint_path,
            dataloaders=dl_test)

if __name__ == '__main__':
    from VAEGAN.vaegan import VAEGAN
    
    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)
    
    # parse training args
    train_args = GenerativeLightningModule.parser_trainer_args(parent_parser)
    
    # add model specific args
    model_args = VAEGAN.parse_model_args(parent_parser)
    
    # add data specific args
    data_args = Era5EobsTopoDataset_v2.parse_data_args(parent_parser)
    
    if train_args.test_version==None:
        GenerativeLightningModule.train_model(train_args, data_args, model_args)
    else:
        GenerativeLightningModule.test_model( train_args, data_args )
    

    




