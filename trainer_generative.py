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
from torchinfo import summary
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
        
        # Load Neural Model

        self.neural_net = neural_net
        
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.register_buffer('target_scale', torch.as_tensor( scaler_target.scale_[0]) )
        
        self.debugging = debugging
        
        self.loss_func = VAEGANLoss( content_loss=False, 
                                content_loss_name='ensmeanMSE_phys',
                                kl_weight=1e-5, 
                                # cl_weight=0.2  
                                )
        
        self.fid_loss = tm.image.fid.FrechetInceptionDistance( normalize=True )
        # self.fid_loss.to(self.device)
                   
    def forward(self, variable_fields, constant_fields, image):
        
        score , image, z_mean, z_logvars = self.neural_net(variable_fields, constant_fields, image)
        
        return score , image, z_mean, z_logvars
            
    def step(self, batch:Dict, step_name, optimizer_idx=0):

        # Extracting data from batch
        variable_fields = batch['fields'].to(self.dtype)
        constant_fields = batch['topo'].to(self.dtype)
        image_scaled = batch['rain'].to(self.dtype) # Images set to (c,h,w) == 0.0 correspond to inputs that arent real
                                # False datums will have corresponding images of all 0
        mask = batch['mask']
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        mask = ~mask    
        
        if optimizer_idx == 0:
            # generator step
            
            # li_score_pred, li_image, li_zmean, li_zlogvars  = self.forward( variable_fields, constant_fields, image_scaled  )
            score_pred , image_output_scaled, z_mean, z_logvar = self.forward( variable_fields, constant_fields, image_scaled  )
            
            # score_pred = torch.stack(li_score_pred, dim=-1)
            # image = torch.stack(li_image, dim=-1)
            # zmean = torch.stack(li_zmean, dim=-1)
            # z_logvar = torch.stack(li_zlogvars, dim=-1)
            
            loss = self.loss_func( -torch.ones_like(score_pred), score_pred, z_mean, z_logvar, mask,
                            self.neural_net, constant_fields  )
            
            # Prediction metrics for the generated images                            
            output = {'img_pred_mse': F.mse_loss(
                torch.masked_select(image_scaled, mask),
                torch.masked_select(image_output_scaled.squeeze(1), mask) )}
                    
            output['loss'] = loss
        
        elif optimizer_idx == 1:
            # Discrimintor step
            
        
            # Real images
            _ = einops.rearrange( variable_fields, '(b t) h w d ->b (d t) h w', t=4)
            _ = self.neural_net.grouped_conv2d_reduce_time(_)
            
            score_pred  = self.neural_net.discriminator( image_scaled[:,None], _, constant_fields[:, None]  )
            loss1 = self.loss_func( torch.ones_like(score_pred), score_pred, z_mean=None, z_logvar=None, mask=None)
            
            # Fake images
            score_pred_fake , image_output_scaled, z_mean, z_logvar = self.forward( variable_fields, constant_fields, image_scaled  )
            loss2 = self.loss_func(-torch.ones_like(score_pred), score_pred_fake, z_mean, z_logvar, mask) 
            loss = loss1 + loss2
            output = {'loss':loss}
        return output

    def training_step(self, batch, batch_idx, optimizer_idx ) :
        # training_step defines the train loop. It is independent of forward
        output = self.step(batch, "train", optimizer_idx)
        self.log("train_loss/loss",output['loss'])

        if 'img_pred_mse' in output:
            self.log("train/img_mse_rain", 
                        output['img_pred_mse'],
                        on_step=False, on_epoch=True )    
    
        return output
    
    def validation_step(self, batch, batch_idx):
        
        # During validation we calculate the fid score on a set of generated images
        
        # Extracting data from batch
        variable_fields = batch['fields'].to(self.dtype)
        constant_fields = batch['topo'].to(self.dtype)
        image_scaled = batch['rain'].to(self.dtype) # Images set to (c,h,w) == 0.0 correspond to inputs that arent real
                                # False datums will have corresponding images of all 0
        mask = batch['mask']
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        mask = ~mask    
        
        image_output_scaled, *_= self.neural_net.generator( variable_fields, constant_fields )
        
        # For FID loss input must be sclaed to 0-1 range
        if self.fid_loss.device != image_output_scaled.device:
            self.fid_loss.to(image_output_scaled.device)
            
        _ = ( image_scaled.shape[0], 3, *image_scaled.shape[1:] )
        self.fid_loss.update(image_scaled[:,None].expand(_)/ self.scaler_target.feature_range[1], real=True )
        self.fid_loss.update(image_output_scaled.expand(_) /self.scaler_target.feature_range[1], real=False)
        
        # self.fid_loss.compute()
        
        return None
    
    def validation_epoch_end(self, validation_step_outputs):
        output = {}
        
        fid_score = self.fid_loss.compute()

        self.log("val_fid", fid_score, prog_bar=True, on_epoch=True)
        self.fid_loss.reset()
        
    def test_step(self, batch, batch_idx) :
        output = {}
               
        # Extracting data from batch
        
        variable_fields = batch['fields'].to(self.dtype)
        constant_fields = batch['topo'].to(self.dtype)
        image_scaled = batch['rain'].to(self.dtype) # Images set to (c,h,w) == 0.0 correspond to inputs that arent real
                                # False datums will have corresponding images of all 0
        mask = batch['mask']
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        mask = ~mask    
        target_date_window = batch.pop('target_date_window', None)
        
        # generating image
        image_output_scaled, *_= self.neural_net.generator( variable_fields, constant_fields )
        
        # Calc FID loss 
        if self.fid_loss.device != image_output_scaled.device:
            self.fid_loss.to(image_output_scaled.device)
        _ = ( image_scaled.shape[0], 3, *image_scaled.shape[1:] )
        self.fid_loss.update(image_scaled[:,None].expand(_)/self.scaler_target.feature_range[1], real=True )
        self.fid_loss.update(image_output_scaled.expand(_)/self.scaler_target.feature_range[1], real=False)
        
        # Scaling up image
        output = {}
        output['pred_rain'] = (image_output_scaled.squeeze().cpu().numpy() / self.scaler_target.scale_)
        output['target_rain'] = (image_scaled.cpu().numpy() / self.scaler_target.scale_)
        output['mask'] = mask.cpu().numpy()
        output['target_date_window'] = target_date_window
        return output
    
    def test_epoch_end(self, outputs) -> None:
        print("\nSaving Test Output to File")

        # Concatenating outputs of test steps
        pred_rain = np.concatenate( [d['pred_rain'] for d in outputs] )
        target_rain = np.concatenate( [d['target_rain'] for d in outputs] )
        mask = np.concatenate([d['mask'] for d in outputs])
        target_date_window = np.concatenate( [d['target_date_window'] for d in outputs] )

        # sorting outputs based on dates
        sort_idx = np.argsort(target_date_window)
        pred_rain = pred_rain[sort_idx]
        target_rain = target_rain[sort_idx]
        mask = mask[sort_idx]
        target_date_window = target_date_window[sort_idx]
        
                
        # Saving Model Outputs To file 
        output = {}
        output['pred_rain'] = pred_rain
        output['target_rain'] = target_rain
        output['mask'] = mask
        output['target_date_window'] = target_date_window
        
        
        suffix = f"{self.dconfig.test_start}_{self.dconfig.test_end}"
        file_path = os.path.join( self.logger.log_dir , f"test_output_{suffix}.pkl" ) 
        with open( file_path, "wb") as f:
            pickle.dump( output, f )
            
        # Logging Model Evaluation Metrics
        fid_score = self.fid_loss.compute().cpu().numpy().item()
        
        sqd_diff = (target_rain-pred_rain)**2
        mse_score = sqd_diff[mask].mean()
        r10_mse_score = sqd_diff[ np.logical_and(mask, target_rain>10.0 )  ].mean()
                
        self.fid_loss.reset()
        self.log("test_fid", fid_score, prog_bar=True, on_epoch=True)
        
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
            'test_fid_score': fid_score,
            'test_r10mse': r10_mse_score.item()}

        file_path_summary = os.path.join(self.logger.log_dir, f"summary_{suffix}.yaml")
        with open(file_path_summary, "w") as f:
            yaml.dump( summary, f)
        return super().test_epoch_end(outputs)
    
    
    def configure_optimizers(self):
        #debug
        
        generator_params = (*self.neural_net.encoder.parameters(), *self.neural_net.decoder.parameters() )
        
        optimizer_generator = Adam(  generator_params , lr=5e-6 )
        frequency_generator = 5
        dict_generator = {
            'optimizer': optimizer_generator,
            'frequency': frequency_generator
        }
        
        optimizer_discriminator = Adam(self.neural_net.discriminator.parameters() , lr=5e-6 )
        frequency_discriminator = 1
        dict_discriminator = {
            'optimizer': optimizer_discriminator,
            'frequency': frequency_discriminator,
        }
        
        # lr_scheduler = get_constant_schedule_with_warmup( optimizer, num_warmup_steps=1000, last_epoch=-1)
        
        return (dict_generator, dict_discriminator)
        
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
        
        config_train = yaml.safe_load(open(path_config_train,"r"))
        config_data = yaml.safe_load(open(path_config_data,"r"))
        config_model = yaml.safe_load(open(path_config_model,"r"))
        
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
        
        yaml.dump(train_args, open( os.path.join(dir_, "trainer.yaml"), "w" ) )
        yaml.dump(data_args, open( os.path.join(dir_, "data.yaml"), "w" ) )
        yaml.dump(model_args, open( os.path.join(dir_, "model.yaml"), "w" ) )
        
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
        train_parser.add_argument("--gpus", default=1, type=int)
        # train_parser.add_argument("--sample_size", default=100)
        train_parser.add_argument("--nn_name", default="VAEGAN", choices=["VAEGAN"])
        
        train_parser.add_argument("--max_epochs", default=300, type=int)
        train_parser.add_argument("--batch_size", default=48, type=int)
                
        train_parser.add_argument("--debugging",action='store_true', default=False )
        train_parser.add_argument("--workers", default=6, type=int )
                        
        train_parser.add_argument("--test_version", default=None, type=int, required=False ) 
        train_parser.add_argument("--val_check_interval", default=1.0, type=float)
        train_parser.add_argument("--prefetch", type=int, default=2, help="Number of batches to prefetch" )
                
        train_parser.add_argument("--ckpt_dir", type=str, default='Checkpoints')
        
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
        dir_model = os.path.join(root_dir,f"{train_args.exp_name}/{train_args.nn_name}")
        
        # Adjusting val_check_interval
        # If val_check_interval is a float then it represents proportion of an epoc
        trainer = pl.Trainer(gpus=train_args.gpus,
                            default_root_dir = dir_model,
                            callbacks =[EarlyStopping(monitor="val_fid", patience=25),
                                            ModelCheckpoint(
                                                monitor="val_fid",
                                                filename='{epoch}-{step}-{val_fid:.3f}',
                                                save_last=False,
                                                auto_insert_metric_name=True,
                                                save_top_k=1)],
                            enable_checkpointing=True,
                            precision=32,
                            max_epochs=train_args.max_epochs,
                            num_sanity_val_steps=0,
                            gradient_clip_val=1.5,
                            
                            limit_train_batches=51 if train_args.debugging else None,
                            limit_val_batches=5 if train_args.debugging else None,
                            limit_test_batches=5 if train_args.debugging else None,
                            val_check_interval=None if train_args.debugging else train_args.val_check_interval
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
                                persistent_workers=False)

        dl_val = DataLoader(ds_val, train_args.batch_size, 
                                num_workers=train_args.workers,
                                drop_last=True, 
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn,
                                pin_memory=True,
                                persistent_workers=False)

        dl_test = DataLoader(ds_test, train_args.batch_size, 
                                drop_last=True,
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn,
                                pin_memory=True,
                                persistent_workers=False )
        

        # Define Lightning Module
        glm = GenerativeLightningModule(scaler_features,scaler_target,
                                        neural_net,
                                        train_args.debugging,
                                        dconfig = data_args)

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
                        val_dataloaders=dl_val)
        
        # Test the Trainer
        trainer.test(dataloaders=dl_test, ckpt_path='best')

    @staticmethod
    def test_model( train_args, data_args):

        root_dir = train_args.ckpt_dir if train_args.ckpt_dir else ''
        dir_model = os.path.join(root_dir,f"{train_args.exp_name}/{train_args.nn_name}")
        dir_model_version = os.path.join(dir_model, "lightning_logs",f"version_{train_args.test_version}")
        
        # Load Confifs and Scalers
        train_args, data_args, model_args = GenerativeLightningModule.load_configs( dir_model_version )
        scaler_features, scaler_target = GenerativeLightningModule.load_scalers( dir_model_version )
        
        checkpoint_path = next( ( elem for elem in glob.glob(os.path.join( dir_model_version, "checkpoints", "*")) 
                                    if elem[-4:]=="ckpt"))
        
        neural_net=VAEGAN(**vars(model_args))
        
        trainer = pl.Trainer(  
                    precision=16,
                    enable_checkpointing=True,
                    logger=False,
                    gpus=train_args.gpus,
                    default_root_dir=dir_model_version)

        # Making Dset
        ds_test = Era5EobsTopoDataset_v2( start_date=data_args.test_start, end_date=data_args.test_end,
                                         dconfig=data_args,
                                    xarray_decode=True,
                                    scaler_features=scaler_features, 
                                    scaler_target=scaler_target)
    
        dl_test = DataLoader(ds_test, train_args.batch_size, 
                                drop_last=False,
                                collate_fn=Era5EobsTopoDataset_v2.collate_fn,
                                pin_memory=True,
                                persistent_workers=False )

        # Define Lightning Module
        glm = GenerativeLightningModule(scaler_features,scaler_target, neural_net)
        
        # Test the Trainer
        trainer.test_model(
            glm,
            ckpt_path=checkpoint_path,
            dataloaders=dl_test)

if __name__ == '__main__':
    from vaegan import VAEGAN
    
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
        GenerativeLightningModule.test( train_args, data_args )
    

    




