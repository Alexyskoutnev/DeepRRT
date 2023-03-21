import torch
from torch import nn
import numpy as np 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.device = device


    def forward(self, map):
        flatten_vec = map.flatten()
        flatten_vec = torch.from_numpy(flatten_vec)
        x = self.network(flatten_vec)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.network = nn.Sequential(
                    nn.Linear(input_size, 1280),
                    nn.ReLU(),
                    nn.Linear(1280, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

class Agent(object):
    def __init__(self, map) -> None:
        pass