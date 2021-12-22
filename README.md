<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>

# Neural GLMs

For the moment:

## Tl;dr
Our Framework for training/testing requires 4 elements to be defined:
GLM model
  - Wraps the neural network. Handles GLM related logic such as link functions and distributional losses
  - Example networks GLM, HGLM, DGLM
Neural net e.g. LSTM, FFN, TRUNET
DataLoading e.g. Toy Dataset, Australian Weather Dataset

- Neural GLM model Structure: 
  - Neural Network: any neural network structure for a given task
  - GLM model: Wraps the neural network. Handles GLM related logic such as link functions and distributional losses
- DataLoading
  - Two Evaluation Datasets: 1.Toy Dataset 2.Australian Weather Dataset


# Training and Testing: 
python3 train.py
This will train a model and save predictions.

# Evaluatinos:
Evaluation.ipynb


# Dependencies
The dependencies can be installed with ```pip3 install -r requirements.txt```
```pip3 install git+https://github.com/Rilwan-A/Better_LSTM_PyTorch.git```
```






