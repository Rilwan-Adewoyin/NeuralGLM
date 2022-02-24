#!/usr/bin/env bash
python3 train.py --target_distribution_name lognormal_hurdle --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps_log --mu_params 1.5,0.01 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshiftn --disp_params 0.5,6.0,0.1 --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --pos_weight 1.0 --target_range 0,4 --debugging  --dropoutw 0.85

python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.4,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1.0 --target_range 0,8 --debugging --dropoutw 0.85

python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.1,10.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 2 --max_j 12 --target_range 0,2 --debugging --dropoutw 0.85
python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.5,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --pos_weight 1.0 --cp_version 3 --max_j 12 --target_range 0,6 --debugging --dropoutw 0.85
python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.1,10.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 3 --target_range 0,2 --debugging --dropoutw 0.85
python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.5,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --pos_weight 1.0 --cp_version 5 --j_window_size 3 --target_range 0,6 --debugging --dropoutw 0.85

python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 1.0,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 2 --max_j 12 --target_range 0,2 --debugging --dropoutw 0.85
python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 1.0,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --pos_weight 1.0 --cp_version 3 --max_j 12 --target_range 0,6 --debugging --dropoutw 0.85
python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 1.0,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 3 --target_range 0,2 --debugging --dropoutw 0.85
python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xmult_exponential_yshifteps --mu_params 1.0,0.0001 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --pos_weight 1.0 --cp_version 5 --j_window_size 3 --target_range 0,6 --debugging --dropoutw 0.85
exec bash