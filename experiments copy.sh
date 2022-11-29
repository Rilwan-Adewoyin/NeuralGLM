#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# =====================ROBUSTNESS EXPERIEMENTS)
# 1) Train on all Uk and Test all UK - Train for first 5 years - Test on next 35 years
#Gamma
python3 train.py --train_start 1979 --train_end 1983 --val_start 1983 --val_end 1984-07  --test_start 1984-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 6 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_5y_35y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 1979 --train_end 1983 --val_start 1983 --val_end 1984-07  --test_start 1984-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_5y_35y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1983 --val_start 1983 --val_end 1984-07  --test_start 1984-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_5y_35y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1983 --val_start 1983 --val_end 1984-07  --test_start 1984-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_5y_35y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1983 --val_start 1983 --val_end 1984-07  --test_start 1984-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_5y_35y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1983 --val_start 1983 --val_end 1984-07  --test_start 1984-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_5y_35y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33 &&

# 2) Train on all Uk and Test all UK - Train for first 10 years - Test on next 30 years
#Gamma
python3 train.py --train_start 1979 --train_end 1987-01 --val_start 1987-01 --val_end 1989-07  --test_start 1989-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_10y_30y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 1979 --train_end 1987-01 --val_start 1987-01 --val_end 1989-07  --test_start 1989-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_10y_30y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1987-01 --val_start 1987-01 --val_end 1989-07  --test_start 1989-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_10y_30y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1987-01 --val_start 1987-01 --val_end 1989-07  --test_start 1989-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_10y_30y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1987-01 --val_start 1987-01 --val_end 1989-07  --test_start 1989-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_10y_30y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1987-01 --val_start 1987-01 --val_end 1989-07  --test_start 1989-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_10y_30y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33 &&

# 3) Train on all Uk and Test all UK - Train for first 15 years - Test on next 25 years
## Gamma
python3 train.py --train_start 1979 --train_end 1989-10 --val_start 1989-10 --val_end 1994-07  --test_start 1994-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_15y_25y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 1979 --train_end 1989-10 --val_start 1989-10 --val_end 1994-07  --test_start 1994-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_15y_25y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1989-10 --val_start 1989-10 --val_end 1994-07  --test_start 1994-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_15y_25y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1989-10 --val_start 1989-10 --val_end 1994-07  --test_start 1994-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_15y_25y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1989-10 --val_start 1989-10 --val_end 1994-07  --test_start 1994-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_15y_25y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1989-10 --val_start 1989-10 --val_end 1994-07  --test_start 1994-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_15y_25y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33 &&

# 4) Train on all Uk and Test all UK - Train for first 20 years - Test on next 20 years
## Gamma
python3 train.py --train_start 1979 --train_end 1993-03 --val_start 1995-03 --val_end 1999-07  --test_start 1999-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_20y_20y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 1979 --train_end 1993-03 --val_start 1995-03 --val_end 1999-07  --test_start 1999-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_20y_20y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1993-03 --val_start 1995-03 --val_end 1999-07  --test_start 1999-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_20y_20y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1993-03 --val_start 1995-03 --val_end 1999-07  --test_start 1999-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_20y_20y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1993-03 --val_start 1995-03 --val_end 1999-07  --test_start 1999-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_20y_20y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1993-03 --val_start 1995-03 --val_end 1999-07  --test_start 1999-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_20y_20y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33 &&

# 5) Train on all Uk and Test all UK - Train for first 25 years - Test on next 15 years
## Gamma
python3 train.py --train_start 1979 --train_end 1998 --val_start 1998 --val_end 2004-07  --test_start 2004-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_25y_15y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 1979 --train_end 1998 --val_start 1998 --val_end 2004-07  --test_start 2004-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_25y_15y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1998 --val_start 1998 --val_end 2004-07  --test_start 2004-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_25y_15y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1998 --val_start 1998 --val_end 2004-07  --test_start 2004-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_25y_15y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1998 --val_start 1998 --val_end 2004-07  --test_start 2004-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_25y_15y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 1998 --val_start 1998 --val_end 2004-07  --test_start 2004-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_25y_15y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33 &&

# 6) Train on all Uk and Test all UK - Train for first 30 years - Test on next 10 years
## Gamma
python3 train.py --train_start 1979 --train_end 2002-07 --val_start 2002-07 --val_end 2009-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_30y_10y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 1979 --train_end 2002-07 --val_start 2002-07 --val_end 2009-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_30y_10y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 2002-07 --val_start 2002-07 --val_end 2009-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_30y_10y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 2002-07 --val_start 2002-07 --val_end 2009-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_30y_10y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 2002-07 --val_start 2002-07 --val_end 2009-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_30y_10y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 1979 --train_end 2002-07 --val_start 2002-07 --val_end 2009-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "Robustness_30y_10y" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations All --locations_test All --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33 &&

# 7) Train on all Uk and Test all UK - Train on last 20 years - Test on same 20 years. Train on 50% location Test on different 50% location
## Gamma
python3 train.py --train_start 2009-07 --train_end 2017-07 --val_start 2017-07 --val_end 2019-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "All_UK_traintest_20y_unseenlocation" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations "WholeMapSplit_5_5" --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001  --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --target_range 0,2 --beta1 0.2 --val_check_interval 0.33  &&
# Compound Poisson v15, 9, 38, 12, 25 (iterations of Rilwan version - method 1 and 2)
python3 train.py --train_start 2009-07 --train_end 2017-07 --val_start 2017-07 --val_end 2019-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "All_UK_traintest_20y_unseenlocation" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations "WholeMapSplit_5_5" --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 2009-07 --train_end 2017-07 --val_start 2017-07 --val_end 2019-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "All_UK_traintest_20y_unseenlocation" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations "WholeMapSplit_5_5" --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2  --val_check_interval 0.33 &&
python3 train.py --train_start 2009-07 --train_end 2017-07 --val_start 2017-07 --val_end 2019-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "All_UK_traintest_20y_unseenlocation" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations "WholeMapSplit_5_5" --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 5 --j_window_size 6 --target_range 0,2 --beta1 0.45 --val_check_interval 0.33 &&
python3 train.py --train_start 2009-07 --train_end 2017-07 --val_start 2017-07 --val_end 2019-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "All_UK_traintest_20y_unseenlocation" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations "WholeMapSplit_5_5" --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 3 --max_j 12 --target_range 0,2 --beta1 0.3 --val_check_interval 0.33 &&
python3 train.py --train_start 2009-07 --train_end 2017-07 --val_start 2017-07 --val_end 2019-07  --test_start 2009-07 --test_end 2019-07 --batch_size 96 --gen_size 30 --cache_gen_size 140 --workers 12 --ckpt_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/ --exp_name "All_UK_traintest_20y_unseenlocation" --data_dir /mnt/Data1/akann1w0w1ck/NeuralGLM/Data/uk_rain --gpus 1  --locations "WholeMapSplit_5_5" --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 0.5,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 16 --target_range 0,2  --beta1 0.3 --val_check_interval 0.33


# 8) Do benchmarks from paper
# - IFS 
# - best from the paper
# - compound poisson model from previous paper
# (on experiment 4) 