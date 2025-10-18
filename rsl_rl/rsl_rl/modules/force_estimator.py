import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class Encoder(nn.Module):
    is_recurrent = False
    
    def __init__(self,  num_obs,
                        hidden_dims=[256, 256, 256],
                        feature_dim=128,
                        activation='elu',
                        **kwargs):
        if kwargs:
            print("Encoder.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Encoder, self).__init__()

        activation = get_activation(activation)

        encoder_layers = []
        encoder_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], feature_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"Encoder MLP: {self.encoder}")

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        features = self.encoder(observations)
        return features

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

class ForceEstimator(nn.Module):
    is_recurrent = False
    def __init__(self,  feature_dim=128,
                        estimator_hidden_dims=[256, 256, 256],
                        num_force=9,
                        activation='elu',
                        **kwargs):
        if kwargs:
            print("ForceEstimator.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ForceEstimator, self).__init__()

        activation = get_activation(activation)

        estimator_layers = []
        estimator_layers.append(nn.Linear(feature_dim, estimator_hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(estimator_hidden_dims)):
            if l == len(estimator_hidden_dims) - 1:
                estimator_layers.append(nn.Linear(estimator_hidden_dims[l], num_force*3))
            else:
                estimator_layers.append(nn.Linear(estimator_hidden_dims[l], estimator_hidden_dims[l + 1]))
                estimator_layers.append(activation)
        self.estimator = nn.Sequential(*estimator_layers)

        print(f"Estimator MLP: {self.estimator}")

    def reset(self, dones=None):
        pass

    def forward(self, features):
        forces = self.estimator(features)
        return forces

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
