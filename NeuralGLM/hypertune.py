import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from neural_nets import MAP_NAME_NEURALMODEL
from glms import MAP_NAME_GLM
from dataloaders import MAP_NAME_DSET
from trainer_neuralGLM import parse_train_args

#TODO: Make it run on 2 GPUS

def objective(trial: optuna.Trial) -> float:
    # Setting Up Fixed params    
    li_args = [
                "--exp_name", "hypertune",

                "--locations", "All",
                "--locations_test", "All",

                "--target_distribution_name","compound_poisson",
                "--mu_link_name", "xmult_exponential_yshifteps",
                "--mu_params", "0.5,0.0001",

                "--dispersion_distribution_name", "uniform_positive",
                "--dispersion_link_name", "xshiftn_relu_timesm_yshifteps",
                "--disp_params", "0.5,6.0",
                "--p_link_name", "divn_sigmoid_clampeps_yshiftm",
                "--p_params", "2,1",
                "--j_window_size", "6",
                "--approx_method", "gosper", 
                              
                "--train_start","1979",
                "--train_end","1995-03",

                "--val_start","1995-03",
                "--val_end","1999-07",

                "--test_start","1999",
                "--test_end","2019-07",

                "--target_range", "0,2",

                "--locations", "All",
                "--locations_test", "All",
                
                
                "--gen_size","50",
                "--cache_gen_size","300",
                
                "--gpus","1",

                "--debugging",
                
                "--learning_rate", "0.00001",
                "--beta1", "0.8",
                "--beta2", "0.85",
                "--eps", "1e-8",
                "--weight_decay", "0.0001",


                "--ckpt_dir","/mnt/Data1/akann1w0w1ck/NeuralGLM/",
                "--data_dir", "/mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain",

                "--data_load_method", "xarray_mult_files_on_disk"
                ]


    # Parse the arguments
    train_args = parse_train_args(list_args = li_args)
    model_args = MAP_NAME_NEURALMODEL['HConvLSTM_tdscale'].parse_model_args(list_args = li_args) 
    data_args = MAP_NAME_DSET['uk_rain'].parse_data_args(list_args = li_args)
    glm_args = MAP_NAME_GLM['DGLM'].parse_glm_args(list_args = li_args)

    # suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-1)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    
    # define the model with the suggested hyperparameters
    model = YourModelClass(num_layers=num_layers, dropout=dropout, learning_rate=lr)
    
    # define logger and trainer
    logger = TensorBoardLogger(save_dir='logs/', name='my_model', version=trial.number)
    trainer = Trainer(logger=logger, max_epochs=50, gpus=1)
    
    # fit the model
    trainer.fit(model)
    
    # return the metric to be minimized
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)