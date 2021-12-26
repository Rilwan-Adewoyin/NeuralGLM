<!-- ROADMAP -->
## Roadmap

- [ ] Qaulitative/Quantitative Analysis of performance of LogNoraml Hurdle Model
- [ ] Write up code for Other Distributions to be tested
- [ ] ...
- [ ] Extend work to using more widely used dataset
    - [ ] TRUNET paper dataset??
    - [ ] Or rain prediction dataset


<p align="right">(<a href="#top">back to top</a>)</p>

# Neural GLMs

For the moment:

## Tl;dr
Our Framework for training/testing requires 3 modular elements to be defined:  
GLM model
  - Wraps the neural network. Handles GLM related logic such as link functions and distributional losses
  - Example networks GLM, HGLM, DGLM  
 
- Neural GLM model Structure: 
  - Neural Network: any neural network structure for a given task
  - GLM model: Wraps the neural network. Handles GLM related logic such as link functions and distributional losses
  - LSTM, FFN, TRUNET  

- DataLoading
  - Two Evaluation Datasets: 1.Toy Dataset 2.Australian Weather Dataset



# Training and Testing: 
python3 train.py
This will train a model and save predictions.
Arguments:...

# Evaluatinos:
Evaluation.ipynb


# Setup 
 ```
    cat requirements.txt | xargs -n 1 pip install
    pip3 install git+https://github.com/Rilwan-A/Better_LSTM_PyTorch.git
    ```
 Get Australian Rain Dataset from https://www.kaggle.com/jsphyg/weather-dataset-rattle-package/download
 Save it in the following path: Data/australia_rain/weatherAUS.csv relative to where this repository has been cloned






