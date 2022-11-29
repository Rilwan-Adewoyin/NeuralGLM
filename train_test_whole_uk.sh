#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python3 train.py --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "train cities test whole UK" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All_Cities --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0   --target_range 0,2 --val_check_interval 888  &&
python3 train.py --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "train cities test whole UK" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All_Cities --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0   --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 2 --max_j 12 --target_range 0,2 --val_check_interval 888 &&
python3 train.py --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "train cities test whole UK" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All_Cities --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0   --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --val_check_interval 888 &&
python3 train.py --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "train cities test whole UK" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All_Cities --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0   --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 6 --target_range 0,2 --val_check_interval 888
python3 train.py --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "train cities test whole UK" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All_Cities --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0   --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --val_check_interval 888